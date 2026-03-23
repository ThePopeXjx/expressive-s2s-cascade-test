#!/usr/bin/env python3
"""Interactive viewer for cascade-test outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Cascade Samples Viewer", layout="wide")


@st.cache_data(show_spinner=False)
def load_records(outputs_dir: str) -> pd.DataFrame:
    root = Path(outputs_dir)
    speech_dir = root / "speech"
    audio_dir = root / "audio"
    transcript_dir = root / "transcript"
    metadata_dir = root / "metadata"

    records: list[dict[str, Any]] = []
    for speech_path in sorted(speech_dir.glob("*.wav")):
        item_id = speech_path.stem
        source_audio = audio_dir / f"{item_id}.wav"
        if not source_audio.exists():
            continue
        transcript_json = transcript_dir / f"{item_id}.json"
        metadata_json = metadata_dir / f"{item_id}.json"

        source_text = ""
        target_transcript = ""
        speaker_id = ""
        style = ""

        if metadata_json.exists():
            try:
                metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
                source_text = str(metadata.get("text", ""))
                speaker_id = str(metadata.get("speaker_id", ""))
                style = str(metadata.get("style", ""))
            except Exception:
                pass

        if transcript_json.exists():
            try:
                transcript = json.loads(transcript_json.read_text(encoding="utf-8"))
                target_transcript = str(transcript.get("transcript", ""))
            except Exception:
                pass

        records.append(
            {
                "id": item_id,
                "speaker_id": speaker_id,
                "style": style,
                "source_text": source_text,
                "target_transcript": target_transcript,
                "source_audio_path": str(source_audio),
                "target_speech_path": str(speech_path),
            }
        )

    return pd.DataFrame(records)


def read_audio_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def main() -> None:
    st.title("Cascade S2S Sample Viewer")

    default_outputs = "/home/jiaxingxu/expressive-s2s/cascade-test/outputs"
    outputs_dir = st.sidebar.text_input("Outputs dir", value=default_outputs)

    df = load_records(outputs_dir)
    if df.empty:
        st.warning("No samples found. Only files with outputs/speech/{id}.wav are included.")
        return

    st.sidebar.markdown("### Filters")
    keyword = st.sidebar.text_input("Keyword (id/text/transcript)", value="").strip().lower()

    speakers = sorted([x for x in df["speaker_id"].dropna().unique().tolist() if x])
    styles = sorted([x for x in df["style"].dropna().unique().tolist() if x])

    selected_speakers = st.sidebar.multiselect("Speaker", options=speakers, default=[])
    selected_styles = st.sidebar.multiselect("Style", options=styles, default=[])

    page_size = st.sidebar.selectbox("Page size", options=[10, 20, 50, 100], index=1)

    view = df.copy()

    if selected_speakers:
        view = view[view["speaker_id"].isin(selected_speakers)]
    if selected_styles:
        view = view[view["style"].isin(selected_styles)]

    if keyword:
        mask = (
            view["id"].str.lower().str.contains(keyword, na=False)
            | view["source_text"].str.lower().str.contains(keyword, na=False)
            | view["target_transcript"].str.lower().str.contains(keyword, na=False)
        )
        view = view[mask]

    total = len(view)
    st.caption(f"Matched samples: {total}")

    if total == 0:
        st.info("No sample matches current filters.")
        return

    total_pages = (total + page_size - 1) // page_size
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

    start = (page - 1) * page_size
    end = min(start + page_size, total)
    page_df = view.iloc[start:end]

    st.dataframe(
        page_df[["id", "speaker_id", "style", "source_text", "target_transcript"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.subheader(f"Samples {start + 1}-{end}")

    for _, row in page_df.iterrows():
        with st.expander(f"{row['id']} | speaker={row['speaker_id']} | style={row['style']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Source Text**")
                st.write(row["source_text"])
                st.markdown("**Source Audio**")
                st.audio(read_audio_bytes(row["source_audio_path"]), format="audio/wav")
            with col2:
                st.markdown("**Target Transcript**")
                st.write(row["target_transcript"])
                st.markdown("**Target Speech**")
                st.audio(read_audio_bytes(row["target_speech_path"]), format="audio/wav")


if __name__ == "__main__":
    main()
