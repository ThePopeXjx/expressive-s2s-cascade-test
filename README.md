# Cascade Test Pipeline

This project currently includes 2 steps:

- Step 1 (`Qwen-Omni-30B`): transcribe / translate Expresso audio to Chinese text
- Step 2 (`CosyVoice3`): synthesize Chinese speech with original speaker style/emotion prompt audio

## Project Layout

- `codes/transcribe.py`: Step-1 pipeline
- `codes/tts.py`: Step-2 pipeline
- `scripts/run_transcribe.sh`: Step-1 main runner
- `scripts/run_transcirbe.sh`: Step-1 compatibility entry
- `scripts/run_tts.sh`: Step-2 runner
- `outputs/`:
  - `audio/{id}.wav`
  - `transcript/{id}.json`
  - `metadata/{id}.json`
  - `speech/{id}.wav`
- `logs/`: run logs (`run_transcribe_MMDDHHMM.log`, `run_tts_MMDDHHMM.log`)

## Dependencies

These two steps may need two seperate environments.

For Step-1:

```bash
pip install -r requirements-step1.txt
```

For Step-2, refer to [CosyVoice3 official repository](https://github.com/FunAudioLLM/CosyVoice) for detailed installation guide.

## Step 1 Usage (Transcribe / Translate)

```bash
bash scripts/run_transcirbe.sh
```

Key outputs:

- `outputs/audio/{id}.wav`
- `outputs/transcript/{id}.json`
- `outputs/metadata/{id}.json`

## Step 2 Usage (TTS with Style Prompt)

```bash
bash scripts/run_tts.sh
```

Step-2 input and output:

- Input transcript: `outputs/transcript/{id}.json` (Chinese text in `transcript` field)
- Input prompt audio: `outputs/audio/{id}.wav` (reference speaker style/emotion)
- Output speech: `outputs/speech/{id}.wav`

Implementation detail:

- Step-2 uses `CosyVoice3` `inference_cross_lingual(...)`
- TTS text default format: `You are a helpful assistant.<|endofprompt|>{中文文本}`
- CosyVoice import path is injected at runtime so `codes/tts.py` can import from external `/home/jiaxingxu/CosyVoice`

## Useful Args

Both steps support:

- `--sample-start`, `--sample-end`, `--max-samples`
- Step-2 additionally supports `--max-samples-per-style` (default in `run_tts.sh`: `25`), counted per `(speaker, style)` group parsed from `{id}`
- `--resume` / `--no-resume`
- `--continue-on-error` / `--stop-on-error`
- `--io-workers`
- `--log-level`, `--log-file-prefix`, `--logs-dir`
