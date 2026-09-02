# Evaluation

TTS has **no single accuracy %**. Train-set loss is not a quality score. Clara v1 dropped loss (~5.9 → 4.1) and still sounded worse.

Score the **same holdout prompts** on three systems: teacher wav, Maya LoRA **on**, Maya LoRA **off**. Temperature **0.4**.

Jacqueline holdout IDs: `jacq_0001`, `jacq_0038`, `jacq_0116`, `jacq_0125`, `jacq_0132`. See [Data](data.md).

## Metrics

| Metric | What it measures | How | Pass (LoRA vs base) |
|--------|------------------|-----|---------------------|
| **WER / CER** | Intelligibility (closest thing to word accuracy) | Whisper-transcribe generated wav vs `manifest.json` `speak` (lowercase, strip punct, keep digit words) | Mean WER ≤ 8%; `jacq_0038` ≤ 15% |
| **Speaker cosine** | Voice lock vs teacher | ECAPA or WavLM embedding cosine vs matching teacher wav | Mean ≥ 0.65 **and** clearly above LoRA-off |
| **Duration ratio** | Unhurried FNOL pacing | `generated_sec / teacher_sec` | Mean 0.85–1.20; no clip &lt; 0.6 or &gt; 1.6 |
| **A/B listen** | Timbre, tags, no harshness | Blind: teacher vs LoRA vs base | LoRA preferred as Jacqueline on ≥ 4/5 |

WER = (insertions + deletions + substitutions) / reference word count. Lower is better. Speaker cosine of 1.0 is identical embedding, not identical audio.

## Procedure

1. Synthesize each holdout `maya_text` with the **training** description, LoRA on and off. Save under `outputs/eval/`. Fast path: Gradio on port 7860.
2. ASR: faster-whisper `large-v3` (same engine as `voice_agent`) or `openai-whisper`. Compute WER with `jiwer`.
3. Speaker: SpeechBrain `spkrec-ecapa-voxceleb` (or WavLM) cosine vs `data/jacqueline/wavs/jacq_XXXX.wav`.
4. Optional GPU: load adapter, teacher-force `metadata_holdout.json`, mean CE on SNAC tokens only (prompt labels already `-100`). This is **not** perceptual quality.

Do not train on holdout IDs.

## Failures we already hit

| Failure | Cause |
|---------|--------|
| LoRA-on cosine ≈ LoRA-off; ears agree they match | Too few steps / too little audio (Clara v2, 18 steps) |
| Loss down, metallic / harsh | Rank / epochs too high (Clara v1, r=64) |
| Wrong words, clean voice | Transcript misaligned with wav (early Clara VAD zip-pair bug) |
| Mumbled claim number | Digit readback not in train enough; watch `jacq_0038` |
