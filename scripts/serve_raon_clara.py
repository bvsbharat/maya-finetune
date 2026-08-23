#!/usr/bin/env python3
"""Clara + caller voice clone demo (Raon-OpenTTS-1B) from a GWVA sim run.

Default source run: 71db9e1fa2bf (stereo recording.wav + transcript.json)
  - right channel = Clara / agent
  - left channel  = Alex / caller
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio
from ema_pytorch import EMA
from hydra.utils import get_class
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
RAON_ROOT = Path(os.environ.get("RAON_ROOT", str(Path.home() / "Raon-OpenTTS"))).resolve()
DEFAULT_CKPT_DIR = Path(
    os.environ.get("RAON_CKPT_DIR", str(RAON_ROOT / "checkpoints" / "Raon-OpenTTS-1B"))
)
DEFAULT_VOCAB = Path(os.environ.get("RAON_VOCAB", str(DEFAULT_CKPT_DIR / "vocab.txt")))
DEFAULT_CKPT = Path(os.environ.get("RAON_CKPT", str(DEFAULT_CKPT_DIR / "ema_only.pt")))
DEFAULT_CONFIG = Path(
    os.environ.get("RAON_CONFIG", str(RAON_ROOT / "src" / "f5_tts" / "configs" / "1b.yaml"))
)
DUAL_REF_DIR = Path(
    os.environ.get(
        "DUAL_REF_DIR",
        str(ROOT / "data" / "dual_clone" / "voices"),
    )
)
TEXT_NUM_EMBEDS = int(os.environ.get("RAON_TEXT_NUM_EMBEDS", "5555"))
MIN_CHARS_PER_SEC = float(os.environ.get("RAON_MIN_CHARS_PER_SEC", "13.0"))
DEFAULT_SPEED = float(os.environ.get("RAON_SPEED", "0.9"))
# Quality-oriented sampling (smoother, less noisy than defaults)
NFE_STEP = int(os.environ.get("RAON_NFE_STEP", "64"))
CFG_STRENGTH = float(os.environ.get("RAON_CFG", "1.5"))
CROSS_FADE = float(os.environ.get("RAON_CROSS_FADE", "0.2"))
TARGET_RMS = float(os.environ.get("RAON_TARGET_RMS", "0.08"))
SWAY = float(os.environ.get("RAON_SWAY", "-0.8"))


def _smooth_output(audio: np.ndarray, sr: int) -> np.ndarray:
    """Light denoise + soft normalize to reduce hiss/harshness."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size < sr // 4:
        return x
    # high-pass rumble
    rc = 1.0 / (2 * np.pi * 70.0)
    dt = 1.0 / sr
    alpha = rc / (rc + dt)
    y = np.empty_like(x)
    prev_x = float(x[0])
    prev_y = 0.0
    for i, xi in enumerate(x):
        yi = alpha * (prev_y + float(xi) - prev_x)
        y[i] = yi
        prev_x, prev_y = float(xi), yi
    try:
        import noisereduce as nr

        y = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.35)
    except Exception:
        pass
    # mild de-ess / soft clip
    rms = float(np.sqrt(np.mean(y**2) + 1e-12))
    if rms > 1e-6:
        y = y * (0.08 / rms)
    y = np.tanh(y * 1.15) / np.tanh(1.15)
    peak = float(np.max(np.abs(y)) + 1e-8)
    if peak > 0.95:
        y = y * (0.95 / peak)
    return y.astype(np.float32)


def _patch_torchaudio_load() -> None:
    def load_sf(path, *args, **kwargs):
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(data.T), sr

    torchaudio.load = load_sf  # type: ignore[assignment]


def _ensure_raon_on_path() -> None:
    src = RAON_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def load_speakers(ref_dir: Path) -> dict[str, dict]:
    """Map display name -> {wav, text, role, label}."""
    refs_path = ref_dir / "refs.json"
    speakers: dict[str, dict] = {}
    if refs_path.exists():
        data = json.loads(refs_path.read_text())
        for sp in data.get("speakers") or []:
            wav = ref_dir / sp["wav"]
            txt_file = ref_dir / f"ref_{sp['role']}.txt"
            text = sp.get("text") or (txt_file.read_text().strip() if txt_file.exists() else "")
            name = sp.get("display_name") or sp.get("label") or sp["role"]
            speakers[name] = {
                "wav": str(wav),
                "text": text,
                "role": sp["role"],
                "label": sp.get("label", sp["role"]),
                "duration_sec": sp.get("duration_sec"),
            }
    # Fallbacks if refs.json missing
    if not speakers:
        for role, name in [("agent", "Clara (agent)"), ("caller", "Alex / caller")]:
            wav = ref_dir / f"ref_{role}.wav"
            txt = ref_dir / f"ref_{role}.txt"
            if wav.exists() and txt.exists():
                speakers[name] = {
                    "wav": str(wav),
                    "text": txt.read_text().strip(),
                    "role": role,
                    "label": role,
                }
    if not speakers:
        raise SystemExit(f"No dual-clone refs found in {ref_dir}")
    return speakers


def _duration_seconds_for_text(gen_text: str, speed: float) -> float:
    n = max(len(gen_text.strip()), 1)
    return max(2.5, n / (MIN_CHARS_PER_SEC * max(speed, 0.3)))


class RaonClaraEngine:
    def __init__(self, config: Path, ckpt: Path, vocab: Path):
        _patch_torchaudio_load()
        _ensure_raon_on_path()
        os.chdir(RAON_ROOT)

        from f5_tts.infer.utils_infer import infer_process, load_vocoder
        from f5_tts.model.cfm import CFM
        from f5_tts.model.utils import get_tokenizer

        self.infer_process = infer_process
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_cfg = OmegaConf.load(str(config))
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        mel_spec_cfg = model_cfg.model.mel_spec
        arch = OmegaConf.to_container(model_cfg.model.arch, resolve=True)

        vocab_char_map, vocab_size = get_tokenizer(str(vocab), "custom")
        print(f"tokenizer_size={vocab_size} text_num_embeds={TEXT_NUM_EMBEDS}", flush=True)
        model = CFM(
            transformer=model_cls(
                **arch,
                text_num_embeds=TEXT_NUM_EMBEDS,
                mel_dim=mel_spec_cfg.n_mel_channels,
            ),
            mel_spec_kwargs=OmegaConf.to_container(mel_spec_cfg, resolve=True),
            vocab_char_map=vocab_char_map,
        ).to(self.device)

        ema = EMA(model, include_online_model=False).to(self.device)
        print(f"Loading Raon checkpoint: {ckpt}", flush=True)
        blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        if "ema_model_state_dict" not in blob:
            raise RuntimeError(f"Checkpoint missing ema_model_state_dict: {ckpt}")
        ema.load_state_dict(blob["ema_model_state_dict"])
        for key, param in ema.ema_model.state_dict().items():
            model.state_dict()[key].copy_(param)
        model.eval()
        self.model = model
        self.model_cfg = model_cfg

        vocoder_dir = RAON_ROOT / "pretrained_models" / "tts-hifigan-libritts-16kHz"
        self.vocoder = load_vocoder(
            vocoder_name=mel_spec_cfg.mel_spec_type,
            is_local=True,
            local_path=str(vocoder_dir) if vocoder_dir.exists() else None,
            device=self.device,
        )
        print("Raon-OpenTTS-1B ready (dual speaker)", flush=True)

    @torch.inference_mode()
    def synthesize(
        self,
        ref_wav: str,
        ref_text: str,
        gen_text: str,
        speed: float = DEFAULT_SPEED,
    ) -> tuple[int, object]:
        ref_text = (ref_text or "").strip().lower()
        gen_text = (gen_text or "").strip()
        speed = float(speed or DEFAULT_SPEED)
        if not ref_wav or not Path(ref_wav).exists():
            raise gr.Error("Reference WAV missing")
        if not ref_text:
            raise gr.Error("Reference transcript is required for cloning")
        if not gen_text:
            raise gr.Error("Enter text to speak")

        fix_duration = _duration_seconds_for_text(gen_text, speed)
        print(
            f"gen_chars={len(gen_text)} speed={speed:.2f} fix_duration_s={fix_duration:.2f} "
            f"nfe={NFE_STEP} cfg={CFG_STRENGTH}",
            flush=True,
        )
        audio, sr, _ = self.infer_process(
            ref_wav,
            ref_text,
            gen_text.lower(),
            self.model,
            self.vocoder,
            mel_spec_type=self.model_cfg.model.mel_spec.mel_spec_type,
            device=self.device,
            speed=speed,
            fix_duration=fix_duration,
            nfe_step=NFE_STEP,
            cfg_strength=CFG_STRENGTH,
            cross_fade_duration=CROSS_FADE,
            target_rms=TARGET_RMS,
            sway_sampling_coef=SWAY,
        )
        audio = _smooth_output(np.asarray(audio, dtype=np.float32), int(sr))
        out_dir = ROOT / "outputs" / "raon-clara"
        out_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_dir / "last_clone.wav"), audio, sr)
        print(f"generated {len(audio)/sr:.2f}s @ {sr}Hz (smoothed)", flush=True)
        return int(sr), audio


def build_ui(engine: RaonClaraEngine, speakers: dict[str, dict], ref_dir: Path) -> gr.Blocks:
    # Prefer Bharat as default when available
    names = list(speakers.keys())
    if "Bharat" in speakers:
        names = ["Bharat"] + [n for n in names if n != "Bharat"]
    default = "Bharat" if "Bharat" in speakers else names[0]
    sp0 = speakers[default]

    examples = {
        "Clara (agent)": (
            "Hi, this is Clara from Guidewire. Thanks for calling — how can I help you today?"
        ),
        "Alex / caller": (
            "Hi Clara, I’m Alex Chen. I’m exploring Agentic FNOL for our personal auto demo."
        ),
        "Bharat": (
            "Hey Sanjay, I started implementing the simulation scenarios we discussed, "
            "and I can walk you through the full flow with personas, evals, and metrics."
        ),
    }

    with gr.Blocks(title="Voice clone · Raon-OpenTTS-1B") as demo:
        gr.Markdown(
            "# Multi-voice clone (Raon-OpenTTS-1B)\n"
            "- **Bharat** (default) from Zoom recording + VTT\n"
            "- **Clara (agent)** / **Alex (caller)** from GWVA run `71db9e1fa2bf`\n"
            "- **Mic / live record:** use the **HTTPS share link** (or `localhost` via SSH tunnel). "
            "Browsers block microphone on plain `http://` remote IPs.\n"
            "Model: [KRAFTON/Raon-OpenTTS-1B](https://huggingface.co/KRAFTON/Raon-OpenTTS-1B)"
        )
        with gr.Row():
            with gr.Column():
                speaker = gr.Dropdown(choices=names, value=default, label="Voice to clone")
                ref_audio = gr.Audio(
                    label="Reference audio (upload or record mic · ~8–15s clean speech)",
                    type="filepath",
                    sources=["upload", "microphone"],
                    interactive=True,
                    editable=True,
                    recording=False,
                    format="wav",
                    value=sp0["wav"] if Path(sp0["wav"]).exists() else None,
                )
                gr.Markdown(
                    "_After recording: paste **exactly what you said** into Reference transcript below._"
                )
                ref_box = gr.Textbox(label="Reference transcript", value=sp0["text"], lines=4)
                gen_box = gr.Textbox(
                    label="Text to speak in that voice",
                    value=examples.get(default, "Hello, this is a voice clone test."),
                    lines=4,
                )
                speed = gr.Slider(
                    minimum=0.5,
                    maximum=1.2,
                    value=DEFAULT_SPEED,
                    step=0.05,
                    label="Speed (lower = longer / slower)",
                )
                btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                out = gr.Audio(label="Cloned speech", type="numpy", interactive=False)
                dur = gr.Textbox(label="Length", interactive=False)

        def on_speaker(name: str):
            sp = speakers[name]
            wav = sp["wav"] if Path(sp["wav"]).exists() else None
            return wav, sp["text"], examples.get(name, "Hello, this is a voice clone test.")

        speaker.change(on_speaker, inputs=[speaker], outputs=[ref_audio, ref_box, gen_box])

        def _run(ref_path, ref_txt, gen_txt, spd):
            sr, audio = engine.synthesize(ref_path, ref_txt, gen_txt, spd)
            return (sr, audio), f"{len(audio)/sr:.2f} seconds @ {sr} Hz"

        btn.click(_run, inputs=[ref_audio, ref_box, gen_box, speed], outputs=[out, dur])

    # Keep allowed paths on the blocks object for launch()
    demo._clone_allowed_paths = [
        str(ref_dir.resolve()),
        str((ROOT / "data").resolve()),
        str((ROOT / "outputs").resolve()),
        "/tmp",
    ]
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--ref-dir", type=Path, default=DUAL_REF_DIR)
    parser.add_argument(
        "--share",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("RAON_SHARE", "1") not in ("0", "false", "False"),
        help="HTTPS gradio.live link (needed for browser mic on remote hosts)",
    )
    args = parser.parse_args()

    for p, name in [(args.config, "config"), (args.ckpt, "ckpt"), (args.vocab, "vocab")]:
        if not p.exists():
            raise SystemExit(f"Missing {name}: {p}")

    speakers = load_speakers(args.ref_dir)
    print("Speakers:", {k: v.get("duration_sec") for k, v in speakers.items()}, flush=True)
    engine = RaonClaraEngine(args.config, args.ckpt, args.vocab)
    demo = build_ui(engine, speakers, args.ref_dir)
    allowed = getattr(demo, "_clone_allowed_paths", [str(args.ref_dir.resolve())])
    print(f"Launching share={args.share} (mic needs HTTPS or localhost)", flush=True)
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=allowed,
        show_error=True,
    )


if __name__ == "__main__":
    main()
