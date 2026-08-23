"""LiveKit cascading voice agent: Local STT -> Ollama LLM -> streaming Maya TTS."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, metrics
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import openai, silero

from plugins.local_stt import LocalSTT, LocalSTTConfig
from plugins.maya_tts import MayaTTS, MayaTTSConfig

ROOT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clara_agent")


def load_cfg() -> dict:
    path = Path(os.environ.get("VOICE_AGENT_CONFIG", ROOT / "config.yaml"))
    with path.open() as f:
        return yaml.safe_load(f)


CFG = load_cfg()
os.environ.setdefault("LIVEKIT_URL", CFG.get("livekit_url", "ws://127.0.0.1:7880"))
os.environ.setdefault("LIVEKIT_API_KEY", CFG.get("livekit_api_key", "devkey"))
os.environ.setdefault("LIVEKIT_API_SECRET", CFG.get("livekit_api_secret", "secret"))


class ClaraAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=CFG["agent_instructions"])


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    stt = LocalSTT(
        LocalSTTConfig(
            engine=CFG.get("stt_engine", "faster_whisper"),
            whisper_model=CFG.get("whisper_model", "large-v3"),
            device=CFG.get("whisper_device", "cuda"),
            compute_type=CFG.get("whisper_compute_type", "float16"),
        )
    )
    llm = openai.LLM(
        model=CFG.get("ollama_model", "gpt-4o:latest"),
        base_url=CFG.get("ollama_base_url", "http://127.0.0.1:11434/v1"),
        api_key=CFG.get("ollama_api_key", "ollama"),
        temperature=0.4,
    )
    tts = MayaTTS(
        MayaTTSConfig(
            model_id=CFG["maya_model_id"],
            lora_path=CFG.get("maya_lora_path"),
            use_lora=bool(CFG.get("maya_use_lora", True)),
            snac_id=CFG["maya_snac_id"],
            description=CFG["maya_description"],
            temperature=float(CFG.get("maya_temperature", 0.4)),
            stream_frames=int(CFG.get("maya_stream_frames", 8)),
        )
    )

    session = AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
        vad=silero.VAD.load(),
        preemptive_generation=True,
    )

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent) -> None:
        m = ev.metrics
        name = type(m).__name__
        parts = [f"[METRICS] type={name}"]
        for key in ("ttfb", "ttft", "duration", "audio_duration", "end_of_utterance_delay"):
            if hasattr(m, key):
                val = getattr(m, key)
                if val is None:
                    continue
                ms = float(val) * 1000.0 if float(val) < 100 else float(val)
                parts.append(f"{key}_ms={ms:.1f}")
        print(" ".join(parts), flush=True)
        metrics.log_metrics(m)

    await session.start(agent=ClaraAgent(), room=ctx.room)
    await session.generate_reply(
        instructions="Greet the user briefly as Clara and ask how you can help with the FNOL waitlist."
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="clara-maya-agent"))
