#!/usr/bin/env python3
"""Step-2 TTS pipeline: synthesize Chinese speech with IndexTTS2."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

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
    speech_indextts2_dir: Path
    logs_dir: Path


@dataclass
class Item:
    item_id: str
    transcript_path: Path
    prompt_audio_path: Path
    output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Chinese speech from transcripts with IndexTTS2."
    )

    parser.add_argument("--indextts2-root", default="/home/jiaxingxu/index-tts")
    parser.add_argument(
        "--indextts2-model-dir",
        default="/mnt/data1/jiaxingxu/pretrained_models/IndexTTS-2",
        help="IndexTTS2 checkpoints directory.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default="/mnt/data1/jiaxingxu/.cache/huggingface",
        help="HuggingFace cache dir used by IndexTTS2 runtime downloads.",
    )
    parser.add_argument(
        "--indextts2-cfg-path",
        default="",
        help="Optional explicit config path. Default: <indextts2-model-dir>/config.yaml",
    )
    parser.add_argument("--use-fp16", action="store_true", default=False)
    parser.add_argument("--use-cuda-kernel", action="store_true", default=False)
    parser.add_argument("--use-deepspeed", action="store_true", default=False)

    parser.add_argument(
        "--output-dir",
        default="/home/jiaxingxu/expressive-s2s/cascade-test/outputs",
    )
    parser.add_argument("--transcript-dir", default=None)
    parser.add_argument("--prompt-audio-dir", default=None)
    parser.add_argument("--speech-indextts2-dir", default=None)

    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--max-samples-per-style",
        type=int,
        default=None,
        help="Limit synthesis count for each style parsed from id.",
    )

    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")

    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--stop-on-error", dest="continue_on_error", action="store_false")

    parser.add_argument("--io-workers", type=int, default=4)

    parser.add_argument("--logs-dir", default="/home/jiaxingxu/expressive-s2s/cascade-test/logs")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file-prefix", default="run_tts_indextts2")

    return parser.parse_args()


def setup_paths(args: argparse.Namespace) -> Paths:
    output_root = Path(args.output_dir)
    transcript_dir = Path(args.transcript_dir) if args.transcript_dir else output_root / "transcript"
    prompt_audio_dir = Path(args.prompt_audio_dir) if args.prompt_audio_dir else output_root / "audio"
    speech_indextts2_dir = (
        Path(args.speech_indextts2_dir)
        if args.speech_indextts2_dir
        else output_root / "speech_indextts2"
    )
    logs_dir = Path(args.logs_dir)

    output_root.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    prompt_audio_dir.mkdir(parents=True, exist_ok=True)
    speech_indextts2_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return Paths(
        output_root=output_root,
        transcript_dir=transcript_dir,
        prompt_audio_dir=prompt_audio_dir,
        speech_indextts2_dir=speech_indextts2_dir,
        logs_dir=logs_dir,
    )


def setup_logger(logs_dir: Path, prefix: str, level: str) -> tuple[logging.Logger, Path]:
    timestamp = datetime.now(ET_TZ).strftime("%m%d%H%M")
    log_path = logs_dir / f"{prefix}_{timestamp}.log"

    logger = logging.getLogger("tts_indextts2")
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


def setup_indextts2_import(indextts2_root: Path) -> None:
    root = indextts2_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_indextts2(args: argparse.Namespace, logger: logging.Logger) -> Any:
    setup_indextts2_import(Path(args.indextts2_root))

    # IndexTTS2 upstream sets HF_HUB_CACHE to a relative './checkpoints/hf_cache'
    # at import time. We explicitly override both env and huggingface_hub constants
    # so cache artifacts stay under a user-specified absolute cache directory.
    hf_cache_dir = Path(args.hf_cache_dir).expanduser().resolve()
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = str(hf_cache_dir)
    os.environ.setdefault("HF_HOME", str(hf_cache_dir.parent))

    try:
        from indextts.infer_v2 import IndexTTS2  # type: ignore
        import huggingface_hub.constants as hf_constants

        hf_constants.HF_HUB_CACHE = str(hf_cache_dir)
        if hasattr(hf_constants, "HUGGINGFACE_HUB_CACHE"):
            hf_constants.HUGGINGFACE_HUB_CACHE = str(hf_cache_dir)
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            f"Failed to import IndexTTS2 from root={args.indextts2_root}. "
            "Please verify repository path and dependencies (uv sync)."
        ) from exc

    model_dir = Path(args.indextts2_model_dir)
    cfg_path = Path(args.indextts2_cfg_path) if args.indextts2_cfg_path else model_dir / "config.yaml"

    if not model_dir.exists():
        raise FileNotFoundError(f"--indextts2-model-dir not found: {model_dir}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"IndexTTS2 config file not found: {cfg_path}")

    logger.info(
        "Loading IndexTTS2 model (root=%s, model_dir=%s, cfg_path=%s, use_fp16=%s, use_cuda_kernel=%s, use_deepspeed=%s)",
        args.indextts2_root,
        model_dir,
        cfg_path,
        args.use_fp16,
        args.use_cuda_kernel,
        args.use_deepspeed,
    )
    logger.info("HuggingFace cache dir=%s", hf_cache_dir)

    return IndexTTS2(
        cfg_path=str(cfg_path),
        model_dir=str(model_dir),
        use_fp16=args.use_fp16,
        use_cuda_kernel=args.use_cuda_kernel,
        use_deepspeed=args.use_deepspeed,
    )


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
        output_path = paths.speech_indextts2_dir / f"{item_id}.wav"
        items.append(
            Item(
                item_id=item_id,
                transcript_path=path,
                prompt_audio_path=prompt_audio_path,
                output_path=output_path,
            )
        )

    logger.info(
        "Prepared %d items (transcript_dir=%s, prompt_audio_dir=%s, speech_indextts2_dir=%s)",
        len(items),
        paths.transcript_dir,
        paths.prompt_audio_dir,
        paths.speech_indextts2_dir,
    )
    return items


def read_transcript(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    text = data.get("transcript", "")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Invalid or empty 'transcript' in {path}")
    return text.strip()


def synthesize_indextts2(indextts2: Any, text: str, prompt_wav: Path, output_path: Path) -> None:
    # Keep speaker and emotion reference aligned with original source audio for
    # style-consistent generation (parallel to CosyVoice3 approach).
    indextts2.infer(
        spk_audio_prompt=str(prompt_wav),
        text=text,
        output_path=str(output_path),
        emo_audio_prompt=str(prompt_wav),
        verbose=False,
    )


def move_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.replace(dst)


def main() -> None:
    args = parse_args()
    paths = setup_paths(args)
    logger, log_path = setup_logger(paths.logs_dir, args.log_file_prefix, args.log_level)

    logger.info("Starting TTS IndexTTS2 pipeline.")
    logger.info("Log file: %s", log_path)

    indextts2 = load_indextts2(args, logger)
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

                text = read_transcript(item.transcript_path)
                tmp_output = item.output_path.with_suffix(".tmp.wav")
                if tmp_output.exists():
                    tmp_output.unlink()

                synthesize_indextts2(
                    indextts2=indextts2,
                    text=text,
                    prompt_wav=item.prompt_audio_path,
                    output_path=tmp_output,
                )

                if not tmp_output.exists():
                    raise RuntimeError(
                        f"IndexTTS2 did not generate output file for id={item.item_id}: {tmp_output}"
                    )

                io_futures.append(submit_pool.submit(move_file, tmp_output, item.output_path))
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
