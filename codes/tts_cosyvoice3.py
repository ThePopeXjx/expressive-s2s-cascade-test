#!/usr/bin/env python3
"""Step-2 TTS pipeline: synthesize Chinese speech with CosyVoice3."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import torch
import torchaudio
from tqdm import tqdm


ET_TZ = ZoneInfo("America/New_York")


class EasternTimeFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=ET_TZ)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()


@dataclass
class Paths:
    output_root: Path
    transcript_dir: Path
    prompt_audio_dir: Path
    speech_cosyvoice3_dir: Path
    logs_dir: Path


@dataclass
class Item:
    item_id: str
    transcript_path: Path
    prompt_audio_path: Path
    output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Chinese speech from transcripts with CosyVoice3."
    )

    parser.add_argument("--cosyvoice3-root", default="/home/jiaxingxu/CosyVoice")
    parser.add_argument(
        "--cosyvoice3-model-dir",
        default="/mnt/data1/jiaxingxu/pretrained_models/Fun-CosyVoice3-0.5B",
    )

    parser.add_argument(
        "--output-dir",
        default="/home/jiaxingxu/expressive-s2s/cascade-test/outputs",
    )
    parser.add_argument("--transcript-dir", default=None)
    parser.add_argument("--prompt-audio-dir", default=None)
    parser.add_argument("--speech-cosyvoice3-dir", default=None)

    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.<|endofprompt|>",
        help="Prefix added before Chinese transcript for cross-lingual inference.",
    )
    parser.add_argument(
        "--use-system-prompt",
        action="store_true",
        default=True,
        help="Enable system prompt prefix.",
    )
    parser.add_argument(
        "--no-system-prompt",
        dest="use_system_prompt",
        action="store_false",
    )

    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--max-samples-per-style",
        type=int,
        default=None,
        help="Limit synthesis count for each style parsed from id.",
    )

    parser.add_argument("--stream", action="store_true", default=False)

    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")

    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--stop-on-error", dest="continue_on_error", action="store_false")

    parser.add_argument("--io-workers", type=int, default=4)

    parser.add_argument("--logs-dir", default="/home/jiaxingxu/expressive-s2s/cascade-test/logs")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file-prefix", default="run_tts_cosyvoice3")

    return parser.parse_args()


def setup_paths(args: argparse.Namespace) -> Paths:
    output_root = Path(args.output_dir)
    transcript_dir = Path(args.transcript_dir) if args.transcript_dir else output_root / "transcript"
    prompt_audio_dir = Path(args.prompt_audio_dir) if args.prompt_audio_dir else output_root / "audio"
    speech_cosyvoice3_dir = (
        Path(args.speech_cosyvoice3_dir)
        if args.speech_cosyvoice3_dir
        else output_root / "speech_cosyvoice3"
    )
    logs_dir = Path(args.logs_dir)

    output_root.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    prompt_audio_dir.mkdir(parents=True, exist_ok=True)
    speech_cosyvoice3_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return Paths(
        output_root=output_root,
        transcript_dir=transcript_dir,
        prompt_audio_dir=prompt_audio_dir,
        speech_cosyvoice3_dir=speech_cosyvoice3_dir,
        logs_dir=logs_dir,
    )


def setup_logger(logs_dir: Path, prefix: str, level: str) -> tuple[logging.Logger, Path]:
    timestamp = datetime.now(ET_TZ).strftime("%m%d%H%M")
    log_path = logs_dir / f"{prefix}_{timestamp}.log"

    logger = logging.getLogger("tts_cosyvoice3")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = EasternTimeFormatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger, log_path


def setup_cosyvoice3_import(cosyvoice3_root: Path) -> None:
    root = cosyvoice3_root.resolve()
    matcha_path = root / "third_party" / "Matcha-TTS"

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if matcha_path.exists() and str(matcha_path) not in sys.path:
        sys.path.insert(0, str(matcha_path))


def load_cosyvoice3(cosyvoice3_root: Path, model_dir: Path, logger: logging.Logger):
    setup_cosyvoice3_import(cosyvoice3_root)
    try:
        from cosyvoice.cli.cosyvoice import AutoModel  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            f"Failed to import CosyVoice from root={cosyvoice3_root}. "
            "Please verify CosyVoice repo path and dependencies."
        ) from exc

    logger.info("Loading CosyVoice3 model from %s", model_dir)
    return AutoModel(model_dir=str(model_dir))


def build_items(args: argparse.Namespace, paths: Paths, logger: logging.Logger) -> list[Item]:
    transcript_files = sorted(paths.transcript_dir.glob("*.json"))

    if not transcript_files:
        raise FileNotFoundError(f"No transcript json found under: {paths.transcript_dir}")

    start = max(args.sample_start, 0)
    end = args.sample_end if args.sample_end is not None else len(transcript_files)
    end = min(end, len(transcript_files))
    if start >= end:
        raise ValueError(
            f"Invalid sample range: start={start}, end={end}, total={len(transcript_files)}"
        )

    selected = transcript_files[start:end]
    if args.max_samples_per_style is not None:
        if args.max_samples_per_style < 1:
            raise ValueError("--max-samples-per-style must be >= 1")
        style_counts: dict[str, int] = {}
        filtered: list[Path] = []
        for path in selected:
            parts = path.stem.split("_")
            speaker = parts[0] if len(parts) >= 1 else "unknown_speaker"
            style = "_".join(parts[1:-1]) if len(parts) >= 3 else "unknown_style"
            speaker_style_key = f"{speaker}::{style}"
            current = style_counts.get(speaker_style_key, 0)
            if current >= args.max_samples_per_style:
                continue
            style_counts[speaker_style_key] = current + 1
            filtered.append(path)
        selected = filtered
        logger.info(
            "Applied per-(speaker,style) cap: max_samples_per_style=%d, remaining=%d groups=%d",
            args.max_samples_per_style,
            len(selected),
            len(style_counts),
        )

    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be >= 1")
        selected = selected[: args.max_samples]

    items: list[Item] = []
    for path in selected:
        item_id = path.stem
        prompt_audio_path = paths.prompt_audio_dir / f"{item_id}.wav"
        output_path = paths.speech_cosyvoice3_dir / f"{item_id}.wav"
        items.append(
            Item(
                item_id=item_id,
                transcript_path=path,
                prompt_audio_path=prompt_audio_path,
                output_path=output_path,
            )
        )

    logger.info(
        "Prepared %d items (transcript_dir=%s, prompt_audio_dir=%s, speech_cosyvoice3_dir=%s)",
        len(items),
        paths.transcript_dir,
        paths.prompt_audio_dir,
        paths.speech_cosyvoice3_dir,
    )
    return items


def read_transcript(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    text = data.get("transcript", "")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Invalid or empty 'transcript' in {path}")
    return text.strip()


def build_tts_text(chinese_text: str, system_prompt: str, use_system_prompt: bool) -> str:
    return f"{system_prompt}{chinese_text}" if use_system_prompt else chinese_text


def run_inference_cosyvoice3(cosyvoice3: Any, tts_text: str, prompt_wav: Path, stream: bool) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for out in cosyvoice3.inference_cross_lingual(tts_text, str(prompt_wav), stream=stream):
        speech = out["tts_speech"]
        if not torch.is_tensor(speech):
            speech = torch.tensor(speech)
        chunks.append(speech.detach().cpu())

    if not chunks:
        raise RuntimeError("CosyVoice3 returned no chunks.")

    if len(chunks) == 1:
        return chunks[0]

    return torch.cat(chunks, dim=-1)


def save_wav(path: Path, speech: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), speech, sample_rate)


def main() -> None:
    args = parse_args()
    paths = setup_paths(args)
    logger, log_path = setup_logger(paths.logs_dir, args.log_file_prefix, args.log_level)

    logger.info("Starting TTS CosyVoice3 pipeline.")
    logger.info("Log file: %s", log_path)

    cosyvoice3 = load_cosyvoice3(
        cosyvoice3_root=Path(args.cosyvoice3_root),
        model_dir=Path(args.cosyvoice3_model_dir),
        logger=logger,
    )

    items = build_items(args, paths, logger)

    submit_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.io_workers))
    io_futures: list[concurrent.futures.Future[Any]] = []

    processed = 0
    skipped = 0
    failed = 0

    try:
        progress = tqdm(total=len(items), desc="Synthesizing", unit="sample")
        for item in items:
            if args.resume and item.output_path.exists():
                skipped += 1
                progress.update(1)
                continue

            try:
                if not item.prompt_audio_path.exists():
                    raise FileNotFoundError(f"Missing prompt audio: {item.prompt_audio_path}")

                zh_text = read_transcript(item.transcript_path)
                tts_text = build_tts_text(zh_text, args.system_prompt, args.use_system_prompt)

                speech = run_inference_cosyvoice3(
                    cosyvoice3=cosyvoice3,
                    tts_text=tts_text,
                    prompt_wav=item.prompt_audio_path,
                    stream=args.stream,
                )

                io_futures.append(
                    submit_pool.submit(save_wav, item.output_path, speech, cosyvoice3.sample_rate)
                )
                processed += 1

            except Exception as exc:  # pragma: no cover
                failed += 1
                logger.exception("Failed item id=%s: %s", item.item_id, exc)
                if not args.continue_on_error:
                    raise

            finally:
                progress.update(1)

        progress.close()

        logger.info("Waiting for I/O workers to finish writing wav files...")
        for fut in tqdm(
            concurrent.futures.as_completed(io_futures),
            total=len(io_futures),
            desc="Finalizing writes",
            unit="file",
        ):
            fut.result()

    finally:
        submit_pool.shutdown(wait=True)

    logger.info(
        "Done. processed=%d skipped=%d failed=%d selected=%d",
        processed,
        skipped,
        failed,
        len(items),
    )


if __name__ == "__main__":
    main()
