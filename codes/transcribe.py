#!/usr/bin/env python3
"""Transcribe Expresso audio with Qwen-Omni and export structured artifacts."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

try:
    from qwen_omni_utils import process_mm_info
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing dependency 'qwen-omni-utils'. Please install it before running."
    ) from exc


@dataclass
class Paths:
    root: Path
    audio_dir: Path
    transcript_dir: Path
    metadata_dir: Path
    logs_dir: Path


ET_TZ = ZoneInfo("America/New_York")


class EasternTimeFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=ET_TZ)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe ylacombe/expresso using Qwen-Omni-30B and save outputs."
    )

    parser.add_argument("--dataset-name", default="ylacombe/expresso")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--model-cache-dir",
        default="/mnt/data1/jiaxingxu/.cache/huggingface",
        help="Cache directory for model/processor downloads from Hugging Face Hub.",
    )

    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Local path or HF model id for Qwen-Omni.",
    )
    parser.add_argument(
        "--transcribe-prompt",
        default="Transcribe the speech into plain text.",
        help="Instruction sent together with each audio.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--use-flash-attn2",
        action="store_true",
        help="Enable flash_attention_2 for transformers backend.",
    )

    parser.add_argument(
        "--output-dir",
        default="/home/jiaxingxu/expressive-s2s/cascade-test/outputs",
        help="Root directory that will contain transcript/, metadata/, audio/.",
    )
    parser.add_argument(
        "--logs-dir",
        default="/home/jiaxingxu/expressive-s2s/cascade-test/logs",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file-prefix", default="run_transcribe")

    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip samples whose transcript and metadata outputs already exist.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume behavior and process all selected samples.",
    )
    parser.add_argument(
        "--overwrite-audio",
        action="store_true",
        help="Overwrite existing audio files.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue to next sample when one sample fails.",
    )
    parser.add_argument(
        "--stop-on-error",
        dest="continue_on_error",
        action="store_false",
        help="Stop immediately on first sample failure.",
    )

    parser.add_argument(
        "--io-workers",
        type=int,
        default=4,
        help="Worker threads for concurrent JSON writing.",
    )

    return parser.parse_args()


def ensure_dirs(output_root: Path, logs_dir: Path) -> Paths:
    audio_dir = output_root / "audio"
    transcript_dir = output_root / "transcript"
    metadata_dir = output_root / "metadata"

    output_root.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return Paths(
        root=output_root,
        audio_dir=audio_dir,
        transcript_dir=transcript_dir,
        metadata_dir=metadata_dir,
        logs_dir=logs_dir,
    )


def setup_logger(logs_dir: Path, prefix: str, level: str) -> tuple[logging.Logger, Path]:
    timestamp = datetime.now(ET_TZ).strftime("%m%d%H%M")
    log_path = logs_dir / f"{prefix}_{timestamp}.log"

    logger = logging.getLogger("transcribe")
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


def safe_id(raw_id: Any) -> str:
    text = str(raw_id)
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_audio(path: Path, audio_array: np.ndarray, sr: int) -> None:
    sf.write(str(path), audio_array, sr)


def load_subset(args: argparse.Namespace, logger: logging.Logger):
    logger.info(
        "Loading dataset: name=%s config=%s split=%s",
        args.dataset_name,
        args.dataset_config,
        args.split,
    )
    dataset = load_dataset(
        path=args.dataset_name,
        name=args.dataset_config,
        split=args.split,
        cache_dir=args.cache_dir,
    )

    if "audio" not in dataset.column_names:
        raise ValueError("Dataset must contain an 'audio' column.")

    # Ensure decoded arrays are available in each sample.
    dataset = dataset.cast_column("audio", Audio(decode=True))

    total = len(dataset)
    start = max(args.sample_start, 0)
    end = args.sample_end if args.sample_end is not None else total
    end = min(end, total)

    if start >= end:
        raise ValueError(f"Invalid sample range: start={start}, end={end}, total={total}")

    indices = list(range(start, end))
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be >= 1")
        indices = indices[: args.max_samples]

    subset = dataset.select(indices)
    logger.info(
        "Selected %d samples (dataset_total=%d, start=%d, end=%d, max_samples=%s)",
        len(subset),
        total,
        start,
        end,
        args.max_samples,
    )
    return subset


def load_model(args: argparse.Namespace, logger: logging.Logger):
    model_ref = args.model_path
    model_ref_path = Path(model_ref)
    if model_ref_path.is_absolute() and not model_ref_path.exists():
        raise FileNotFoundError(
            f"--model-path points to a non-existent local path: {model_ref}. "
            "Use a valid local snapshot directory or an HF repo id like "
            "'Qwen/Qwen3-Omni-30B-A3B-Instruct'."
        )

    logger.info(
        "Loading model and processor from %s (model_cache_dir=%s)",
        model_ref,
        args.model_cache_dir,
    )

    model_kwargs: dict[str, Any] = {
        "dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if args.use_flash_attn2:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    if args.model_cache_dir:
        model_kwargs["cache_dir"] = args.model_cache_dir

    try:
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_ref,
            **model_kwargs,
        )
    except ImportError as exc:
        # Graceful fallback when flash-attn is requested but unavailable.
        if args.use_flash_attn2 and "flash_attn" in str(exc):
            logger.warning(
                "FlashAttention2 requested but flash_attn is not installed. "
                "Falling back to default attention implementation."
            )
            model_kwargs.pop("attn_implementation", None)
            model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                model_ref,
                **model_kwargs,
            )
        else:
            raise
    processor_kwargs: dict[str, Any] = {}
    if args.model_cache_dir:
        processor_kwargs["cache_dir"] = args.model_cache_dir
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_ref, **processor_kwargs)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<not-set>")
    logger.info(
        "Model loaded. torch.cuda.is_available=%s, torch.cuda.device_count=%s, CUDA_VISIBLE_DEVICES=%s",
        torch.cuda.is_available(),
        torch.cuda.device_count(),
        visible,
    )
    return model, processor


def decode_response(generated: Any, input_len: int, processor: Qwen3OmniMoeProcessor) -> str:
    if isinstance(generated, str):
        return generated.strip()

    sequences: Any = generated
    if hasattr(generated, "sequences"):
        sequences = generated.sequences
    elif isinstance(generated, dict) and "sequences" in generated:
        sequences = generated["sequences"]
    elif isinstance(generated, (tuple, list)):
        for item in generated:
            if hasattr(item, "sequences"):
                sequences = item.sequences
                break
            if torch.is_tensor(item):
                sequences = item
                break
            if isinstance(item, dict) and "sequences" in item:
                sequences = item["sequences"]
                break
            if isinstance(item, str):
                return item.strip()

    if not torch.is_tensor(sequences):
        raise TypeError(
            f"Unsupported generate output type for decoding: {type(generated)}"
        )

    output_ids = sequences[:, input_len:]
    text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return text.strip()


def transcribe_one(
    model: Qwen3OmniMoeForConditionalGeneration,
    processor: Qwen3OmniMoeProcessor,
    audio_path: Path,
    prompt: str,
    max_new_tokens: int,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(audio_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )

    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    inputs = inputs.to(model_device)
    # Keep integer tensors (e.g., token ids) unchanged, and align floating inputs
    # (e.g., audio features) with model dtype to avoid conv dtype mismatch.
    for key, value in inputs.items():
        if torch.is_tensor(value) and value.is_floating_point() and value.dtype != model_dtype:
            inputs[key] = value.to(dtype=model_dtype)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            thinker_return_dict_in_generate=True,
            thinker_max_new_tokens=max_new_tokens,
            thinker_do_sample=False,
            return_audio=False,
            use_audio_in_video=False,
        )

    return decode_response(generated, inputs["input_ids"].shape[1], processor)


def main() -> None:
    args = parse_args()
    paths = ensure_dirs(Path(args.output_dir), Path(args.logs_dir))
    logger, log_path = setup_logger(paths.logs_dir, args.log_file_prefix, args.log_level)

    logger.info("Starting transcription pipeline.")
    logger.info("Log file: %s", log_path)

    dataset = load_subset(args, logger)
    model, processor = load_model(args, logger)

    submit_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.io_workers))
    io_futures: list[concurrent.futures.Future[Any]] = []

    processed = 0
    skipped = 0
    failed = 0

    try:
        progress = tqdm(total=len(dataset), desc="Transcribing", unit="sample")
        for sample in dataset:
            raw_id = sample.get("id")
            item_id = safe_id(raw_id)

            transcript_path = paths.transcript_dir / f"{item_id}.json"
            metadata_path = paths.metadata_dir / f"{item_id}.json"
            audio_path = paths.audio_dir / f"{item_id}.wav"

            if args.resume and transcript_path.exists() and metadata_path.exists() and audio_path.exists():
                skipped += 1
                progress.update(1)
                continue

            try:
                audio_obj = sample["audio"]
                audio_array = np.asarray(audio_obj["array"], dtype=np.float32)
                sampling_rate = int(audio_obj["sampling_rate"])

                if args.overwrite_audio or not audio_path.exists():
                    write_audio(audio_path, audio_array, sampling_rate)
                else:
                    logger.debug("Audio already exists and overwrite disabled: %s", audio_path)

                transcript = transcribe_one(
                    model=model,
                    processor=processor,
                    audio_path=audio_path,
                    prompt=args.transcribe_prompt,
                    max_new_tokens=args.max_new_tokens,
                )

                now_utc = datetime.now(timezone.utc).isoformat()
                transcript_payload = {
                    "id": raw_id,
                    "transcript": transcript,
                    "model": args.model_path,
                    "prompt": args.transcribe_prompt,
                    "dataset_name": args.dataset_name,
                    "dataset_config": args.dataset_config,
                    "split": args.split,
                    "generated_at_utc": now_utc,
                }
                metadata_payload = {
                    "id": raw_id,
                    "speaker_id": sample.get("speaker_id"),
                    "style": sample.get("style"),
                    "text": sample.get("text"),
                    "dataset_name": args.dataset_name,
                    "dataset_config": args.dataset_config,
                    "split": args.split,
                    "audio_sampling_rate": sampling_rate,
                    "audio_num_samples": int(audio_array.shape[0]),
                    "audio_duration_sec": round(float(audio_array.shape[0]) / sampling_rate, 4)
                    if sampling_rate > 0
                    else None,
                    "exported_at_utc": now_utc,
                }

                io_futures.append(
                    submit_pool.submit(write_json, transcript_path, transcript_payload)
                )
                io_futures.append(
                    submit_pool.submit(write_json, metadata_path, metadata_payload)
                )

                processed += 1

            except Exception as exc:  # pragma: no cover
                failed += 1
                logger.exception("Failed sample id=%s: %s", raw_id, exc)
                if not args.continue_on_error:
                    raise

            finally:
                progress.update(1)

        progress.close()

        logger.info("Waiting for I/O workers to finish writing files...")
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
        len(dataset),
    )


if __name__ == "__main__":
    main()
