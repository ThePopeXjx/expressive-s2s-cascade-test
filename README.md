# Cascade Test Pipeline

This project currently includes 2 steps:

- Step 1 (`Qwen-Omni-30B`): transcribe / translate Expresso audio to Chinese text
- Step 2: synthesize Chinese speech with original speaker style/emotion prompt audio
- Step 2 Approach A: `CosyVoice3`
- Step 2 Approach B: `IndexTTS2`
- Step 2 Approach C: `FishAudio2` (`fish-speech`)
- Viewer (`Streamlit`): browse samples with source/target audio and text
- Static Export: generate shareable offline HTML package

## Project Layout

- `codes/transcribe.py`: Step-1 pipeline
- `codes/tts_cosyvoice3.py`: Step-2 CosyVoice3 pipeline
- `codes/tts_indextts2.py`: Step-2 IndexTTS2 pipeline
- `codes/tts_fishaudio2.py`: Step-2 FishAudio2 pipeline
- `codes/view_samples.py`: sample viewer web app
- `codes/export_static_html.py`: export static HTML package
- `scripts/run_transcribe.sh`: Step-1 main runner
- `scripts/run_tts_cosyvoice3.sh`: Step-2 CosyVoice3 runner
- `scripts/run_tts_indextts2.sh`: Step-2 IndexTTS2 runner
- `scripts/run_tts_fishaudio2.sh`: Step-2 FishAudio2 runner
- `scripts/run_viewer.sh`: viewer runner
- `scripts/run_export_html.sh`: static HTML exporter
- `outputs/`:
  - `audio/{id}.wav`
  - `transcript/{id}.json`
  - `metadata/{id}.json`
  - `speech_cosyvoice3/{id}.wav`
  - `speech_indextts2/{id}.wav`
  - `speech_fishaudio2/{id}.wav`
- `logs/`: run logs (`run_transcribe_MMDDHHMM.log`, `run_tts_cosyvoice3_MMDDHHMM.log`, `run_tts_indextts2_MMDDHHMM.log`, `run_viewer_MMDDHHMM.log`)

## Dependencies

These two steps may need seperate environments.

For Step-1:

```bash
pip install -r requirements-step1.txt
```

For Step-2 CosyVoice3 implementation, refer to [CosyVoice3 official repository](https://github.com/FunAudioLLM/CosyVoice) for detailed installation guide. A conda environment is recommended.

For Step-2 IndexTTS2 implementation, refer to [IndexTTS2 official repository](https://github.com/index-tts/index-tts) for detailed installation guide. An uv environment is recommended.

For Step-2 FishAudio2 implementation, refer to [fish-speech official repository](https://github.com/fishaudio/fish-speech) for detailed installation guide. A conda environment is recommended.

For Viewer:

```bash
pip install -r requirements-viewer.txt
```

## Step 1 Usage (Transcribe / Translate)

```bash
bash scripts/run_transcirbe.sh
```

Key outputs:

- `outputs/audio/{id}.wav`
- `outputs/transcript/{id}.json`
- `outputs/metadata/{id}.json`

## Step 2 Usage (TTS with Style Prompt)

Approach A (CosyVoice3):

```bash
bash scripts/run_tts_cosyvoice3.sh
```

Output speech:

- `outputs/speech_cosyvoice3/{id}.wav`

Approach B (IndexTTS2):

```bash
bash scripts/run_tts_indextts2.sh
```

Output speech:

- `outputs/speech_indextts2/{id}.wav`

Approach C (FishAudio2):

```bash
bash scripts/run_tts_fishaudio2.sh
```

Output speech:

- `outputs/speech_fishaudio2/{id}.wav`

Shared Step-2 input:

- Input transcript: `outputs/transcript/{id}.json` (Chinese text in `transcript` field)
- Input prompt audio: `outputs/audio/{id}.wav` (reference speaker style/emotion)

Implementation detail (CosyVoice3):

- Step-2 uses `CosyVoice3` `inference_cross_lingual(...)`
- TTS text default format: `You are a helpful assistant.<|endofprompt|>{中文文本}`
- CosyVoice import path is injected at runtime so `codes/tts_cosyvoice3.py` can import from external `/home/jiaxingxu/CosyVoice`

Implementation detail (IndexTTS2):

- Step-2 uses `IndexTTS2` Python API from `indextts.infer_v2`
- Inference call pattern follows official repo examples:
- `IndexTTS2(...).infer(spk_audio_prompt=..., text=..., output_path=..., emo_audio_prompt=...)`
- `codes/tts_indextts2.py` injects `--indextts2-root` into `sys.path` for module import

Implementation detail (FishAudio2):

- Step-2 uses fish-speech **command-line inference same-path API** directly (no CLI shell call):
- `init_model(...)` + `load_codec_model(...)` + `encode_audio(...)` + `generate_long(...)` + `decode_to_audio(...)`
- Per sample inference: prompt audio -> VQ prompt tokens, then Chinese target text -> semantic codes -> waveform

## Useful Args

Both steps support:

- `--sample-start`, `--sample-end`, `--max-samples`
- Step-2 additionally supports `--max-samples-per-style` (default in `run_tts_cosyvoice3.sh`: `25`), counted per `(speaker, style)` group parsed from `{id}`
- `--resume` / `--no-resume`
- `--continue-on-error` / `--stop-on-error`
- `--io-workers`
- `--log-level`, `--log-file-prefix`, `--logs-dir`

## Viewer Usage (Remote Server -> Local Browser)

On remote server:

```bash
bash scripts/run_viewer.sh
```

`run_viewer.sh` passes app args to `view_samples.py` (instead of hardcoding in Python), including:

- `--outputs-dir`
- `--speech-cosyvoice3-subdir`
- `--title`

Then on your local machine (new terminal), create SSH tunnel:

```bash
ssh -L 8501:127.0.0.1:8501 <your_user>@<your_server_host>
```

Open in local browser:

```text
http://127.0.0.1:8501
```

## Static HTML Export (Recommended for Sharing)

Generate static package:

```bash
bash scripts/run_export_html.sh
```

Default output directory:

```text
outputs/static_viewer_fishaudio2/
```

`run_export_html.sh` passes these key args:

- `--target-speech-subdir` (e.g. `speech_cosyvoice3`, `speech_indextts2`, `speech_fishaudio2`)
- `--transcript-subdir` (e.g. `transcript`, `transcript-ja`)
- `--asset-mode` (`hardlink` by default)

Default exporter behavior uses `--asset-mode hardlink` to avoid duplicating local disk usage while preparing package.

It contains:

- `index.html`
- `script.js`
- `audio/{id}.wav` (source audio)
- `{target-speech-subdir}/{id}.wav` (target speech)
- `metadata/{id}.json`
- `{transcript-subdir}/{id}.json`

You can zip and send it to collaborators:

```bash
cd outputs
zip -r static_viewer_fishaudio2.zip static_viewer_fishaudio2
```
