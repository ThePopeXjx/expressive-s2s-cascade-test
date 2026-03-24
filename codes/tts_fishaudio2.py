#!/usr/bin/env python3
"""Step-2 TTS pipeline: synthesize Chinese speech with FishAudio2 (Fish-Speech)."""

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

import numpy as np
import soundfile as sf
import torch
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
    metadata_dir: Path
    speech_fishaudio2_dir: Path
    logs_dir: Path


@dataclass
class Item:
    item_id: str
    transcript_path: Path
    metadata_path: Path
    prompt_audio_path: Path
    output_path: Path


@dataclass
class FishAudio2Runtime:
    model: Any
    decode_one_token: Any
    codec: Any
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Chinese speech from transcripts with FishAudio2."
    )

    parser.add_argument("--fishaudio2-root", default="/home/jiaxingxu/fish-speech")
    parser.add_argument(
        "--llama-checkpoint-path",
        default="/home/jiaxingxu/fish-speech/checkpoints/s2-pro",
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        default="/home/jiaxingxu/fish-speech/checkpoints/s2-pro/codec.pth",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--half", action="store_true", default=False)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--warmup", action="store_true", default=True)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")

    parser.add_argument(
        "--output-dir",
        default="/home/jiaxingxu/expressive-s2s/cascade-test/outputs",
    )
    parser.add_argument("--transcript-dir", default=None)
    parser.add_argument("--metadata-dir", default=None)
    parser.add_argument("--prompt-audio-dir", default=None)
    parser.add_argument("--speech-fishaudio2-dir", default=None)

    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--max-samples-per-style",
        type=int,
        default=None,
        help="Limit synthesis count for each (speaker, style) group parsed from id.",
    )

    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--chunk-length", type=int, default=300)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--iterative-prompt", action="store_true", default=True)
    parser.add_argument("--no-iterative-prompt", dest="iterative_prompt", action="store_false")

    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")

    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--stop-on-error", dest="continue_on_error", action="store_false")

    parser.add_argument("--io-workers", type=int, default=4)

    parser.add_argument("--logs-dir", default="/home/jiaxingxu/expressive-s2s/cascade-test/logs")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file-prefix", default="run_tts_fishaudio2")

    return parser.parse_args()


def setup_paths(args: argparse.Namespace) -> Paths:
    output_root = Path(args.output_dir)
    transcript_dir = Path(args.transcript_dir) if args.transcript_dir else output_root / "transcript"
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else output_root / "metadata"
    prompt_audio_dir = Path(args.prompt_audio_dir) if args.prompt_audio_dir else output_root / "audio"
    speech_fishaudio2_dir = (
        Path(args.speech_fishaudio2_dir)
        if args.speech_fishaudio2_dir
        else output_root / "speech_fishaudio2"
    )
    logs_dir = Path(args.logs_dir)

    output_root.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    prompt_audio_dir.mkdir(parents=True, exist_ok=True)
    speech_fishaudio2_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return Paths(
        output_root=output_root,
        transcript_dir=transcript_dir,
        metadata_dir=metadata_dir,
        prompt_audio_dir=prompt_audio_dir,
        speech_fishaudio2_dir=speech_fishaudio2_dir,
        logs_dir=logs_dir,
    )


def setup_logger(logs_dir: Path, prefix: str, level: str) -> tuple[logging.Logger, Path]:
    timestamp = datetime.now(ET_TZ).strftime("%m%d%H%M")
    log_path = logs_dir / f"{prefix}_{timestamp}.log"

    logger = logging.getLogger("tts_fishaudio2")
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


def setup_fishaudio2_import(fishaudio2_root: Path) -> None:
    root = fishaudio2_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def resolve_device(device: str, logger: logging.Logger) -> str:
    if torch.backends.mps.is_available():
        logger.info("mps is available, switch device to mps.")
        return "mps"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        logger.info("xpu is available, switch device to xpu.")
        return "xpu"
    if not torch.cuda.is_available() and device.startswith("cuda"):
        logger.info("CUDA is not available, switch device to cpu.")
        return "cpu"
    return device


def load_fishaudio2(args: argparse.Namespace, logger: logging.Logger) -> FishAudio2Runtime:
    setup_fishaudio2_import(Path(args.fishaudio2_root))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        from fish_speech.models.text2semantic.inference import (  # type: ignore
            decode_to_audio,
            encode_audio,
            generate_long,
            init_model,
            load_codec_model,
        )
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            f"Failed to import FishAudio2 runtime from root={args.fishaudio2_root}. "
            "Please verify fish-speech path and dependencies."
        ) from exc

    llama_ckpt = Path(args.llama_checkpoint_path)
    decoder_ckpt = Path(args.decoder_checkpoint_path)
    if not llama_ckpt.exists():
        raise FileNotFoundError(f"--llama-checkpoint-path not found: {llama_ckpt}")
    if not decoder_ckpt.exists():
        raise FileNotFoundError(f"--decoder-checkpoint-path not found: {decoder_ckpt}")

    device = resolve_device(args.device, logger)
    precision = torch.half if args.half else torch.bfloat16

    logger.info(
        "Loading FishAudio2 models (CLI-path API) (root=%s, llama=%s, decoder=%s, device=%s, half=%s, compile=%s)",
        args.fishaudio2_root,
        llama_ckpt,
        decoder_ckpt,
        device,
        args.half,
        args.compile,
    )

    # Align with command-line inference implementation:
    # 1) init text2semantic model
    # 2) load DAC codec
    # 3) encode reference audio -> prompt tokens
    # 4) generate semantic codes with generate_long
    # 5) decode with decode_to_audio
    model, decode_one_token = init_model(
        checkpoint_path=str(llama_ckpt),
        device=device,
        precision=precision,
        compile=args.compile,
    )
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )
    codec = load_codec_model(
        codec_checkpoint_path=str(decoder_ckpt),
        device=device,
        precision=precision,
    )

    if args.warmup:
        logger.info("Running warmup request...")
        # Warmup with no prompt reference, consistent with CLI behavior.
        generator = generate_long(
            model=model,
            device=device,
            decode_one_token=decode_one_token,
            text="Hello world.",
            num_samples=1,
            max_new_tokens=128,
            top_p=0.7,
            top_k=30,
            repetition_penalty=1.2,
            temperature=0.7,
            compile=args.compile,
            iterative_prompt=args.iterative_prompt,
            chunk_length=200,
            prompt_text=None,
            prompt_tokens=None,
        )
        for _ in generator:
            pass
        logger.info("Warmup complete.")

    return FishAudio2Runtime(
        model=model,
        decode_one_token=decode_one_token,
        codec=codec,
        device=device,
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
        metadata_path = paths.metadata_dir / f"{item_id}.json"
        output_path = paths.speech_fishaudio2_dir / f"{item_id}.wav"
        items.append(
            Item(
                item_id=item_id,
                transcript_path=path,
                metadata_path=metadata_path,
                prompt_audio_path=prompt_audio_path,
                output_path=output_path,
            )
        )

    logger.info(
        "Prepared %d items (transcript_dir=%s, metadata_dir=%s, prompt_audio_dir=%s, speech_fishaudio2_dir=%s)",
        len(items),
        paths.transcript_dir,
        paths.metadata_dir,
        paths.prompt_audio_dir,
        paths.speech_fishaudio2_dir,
    )
    return items


def read_transcript(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    text = data.get("transcript", "")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Invalid or empty 'transcript' in {path}")
    return text.strip()


def read_reference_text(metadata_path: Path) -> str:
    if not metadata_path.exists():
        return ""
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    text = data.get("text", "")
    return text.strip() if isinstance(text, str) else ""


def infer_fishaudio2(
    runtime: FishAudio2Runtime,
    text: str,
    prompt_wav: Path,
    reference_text: str,
    args: argparse.Namespace,
) -> tuple[int, np.ndarray]:
    from fish_speech.models.text2semantic.inference import (  # type: ignore
        decode_to_audio,
        encode_audio,
        generate_long,
    )

    prompt_tokens = encode_audio(prompt_wav, runtime.codec, runtime.device).cpu()
    prompt_text_list = [reference_text] if reference_text else [""]
    prompt_tokens_list = [prompt_tokens]

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    generator = generate_long(
        model=runtime.model,
        device=runtime.device,
        decode_one_token=runtime.decode_one_token,
        text=text,
        num_samples=1,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        compile=args.compile,
        iterative_prompt=args.iterative_prompt,
        chunk_length=args.chunk_length,
        prompt_text=prompt_text_list,
        prompt_tokens=prompt_tokens_list,
    )

    codes: list[torch.Tensor] = []
    for response in generator:
        if response.action == "sample" and response.codes is not None:
            codes.append(response.codes)
        elif response.action == "next":
            break

    if not codes:
        raise RuntimeError("FishAudio2 generated empty semantic codes.")

    merged_codes = torch.cat(codes, dim=1)
    audio = decode_to_audio(merged_codes.to(runtime.device), runtime.codec)
    audio_np = audio.detach().float().cpu().numpy().astype(np.float32)
    sample_rate = int(runtime.codec.sample_rate)
    return sample_rate, audio_np


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, samplerate=sample_rate)


def move_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.replace(dst)


def main() -> None:
    args = parse_args()
    paths = setup_paths(args)
    logger, log_path = setup_logger(paths.logs_dir, args.log_file_prefix, args.log_level)

    logger.info("Starting TTS FishAudio2 pipeline.")
    logger.info("Log file: %s", log_path)

    runtime = load_fishaudio2(args, logger)
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
                reference_text = read_reference_text(item.metadata_path)
                tmp_output = item.output_path.with_suffix(".tmp.wav")
                if tmp_output.exists():
                    tmp_output.unlink()

                sample_rate, audio = infer_fishaudio2(
                    runtime=runtime,
                    text=text,
                    prompt_wav=item.prompt_audio_path,
                    reference_text=reference_text,
                    args=args,
                )
                write_wav(tmp_output, audio, sample_rate)

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
