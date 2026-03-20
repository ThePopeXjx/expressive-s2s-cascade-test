# Cascade Test - Step 1 (Transcribe Expresso with Qwen-Omni-30B)

This stage transcribes `ylacombe/expresso` audio with locally deployed `Qwen/Qwen3-Omni-30B-A3B-Instruct` (2x A6000), and exports three structured outputs:

- `outputs/transcript/{id}.json`: model-generated transcript
- `outputs/metadata/{id}.json`: original metadata (`id`, `speaker_id`, `style`, `text`, etc.)
- `outputs/audio/{id}.wav`: extracted audio

`text` from dataset is treated as metadata only and is never used as transcript input.

## Project Layout

- `codes/transcribe.py`: transcription pipeline (CLI, logging, progress bar, I/O concurrency)
- `scripts/run_transcribe.sh`: main runner
- `scripts/run_transcirbe.sh`: compatibility entry (requested name, forwards to main runner)
- `outputs/`: generated artifacts
- `logs/`: run logs (`run_transcribe_MMDDHHMM.log`)

## Dependencies

Install in your runtime environment (example):

```bash
pip install torch transformers datasets soundfile tqdm qwen-omni-utils
```

You also need local GPU environment ready for Qwen-Omni. The runner sets:

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

## Usage

Main entry:

```bash
bash scripts/run_transcirbe.sh
```

Override any argument at runtime (passed through to Python):

```bash
bash scripts/run_transcirbe.sh --split validation --max-samples 100 --sample-start 50
```

## Important CLI Args

From `codes/transcribe.py`:

- `--dataset-name` (default: `ylacombe/expresso`)
- `--dataset-config` (optional)
- `--split` (default: `train`)
- `--model-path` (default: `Qwen/Qwen3-Omni-30B-A3B-Instruct`)
- `--transcribe-prompt` (default: `Transcribe the speech into plain text.`)
- `--sample-start`, `--sample-end`, `--max-samples`
- `--resume` / `--no-resume`
- `--continue-on-error` / `--stop-on-error`
- `--io-workers` (concurrent file writing workers)
- `--log-level`, `--log-file-prefix`, `--logs-dir`

## Implementation Notes

- Model invocation follows Qwen cookbook style:
  - `Qwen3OmniMoeForConditionalGeneration`
  - `Qwen3OmniMoeProcessor`
  - `process_mm_info(...)`
- Progress bars:
  - sample-level transcription progress
  - final file-write completion progress
- Logging:
  - console + file logging
  - log filename format example: `run_transcribe_03201339.log`
- Resume behavior:
  - when `--resume` is enabled, sample is skipped only if all three files already exist (`transcript`, `metadata`, `audio`).
