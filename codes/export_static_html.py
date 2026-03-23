#!/usr/bin/env python3
"""Export a static HTML package for cascade-test samples."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export static HTML viewer from outputs directory."
    )
    parser.add_argument(
        "--outputs-dir",
        default="/home/jiaxingxu/expressive-s2s/cascade-test/outputs",
        help="Root outputs dir containing audio/ transcript/ metadata/ speech_*/.",
    )
    parser.add_argument(
        "--export-dir",
        default="/home/jiaxingxu/expressive-s2s/cascade-test/outputs/static_viewer_cosyvoice3",
        help="Destination directory for static HTML package.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on exported samples.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing export dir before exporting.",
    )
    parser.add_argument(
        "--asset-mode",
        choices=["copy", "hardlink", "symlink"],
        default="hardlink",
        help="How to place wav assets into export dir.",
    )
    parser.add_argument(
        "--target-speech-subdir",
        default="speech_cosyvoice3",
        help="Target speech subdir under outputs-dir (e.g., speech_cosyvoice3, speech_indextts2).",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_records(
    outputs_dir: Path, target_speech_subdir: str, max_samples: int | None
) -> list[dict[str, Any]]:
    speech_dir = outputs_dir / target_speech_subdir
    audio_dir = outputs_dir / "audio"
    transcript_dir = outputs_dir / "transcript"
    metadata_dir = outputs_dir / "metadata"

    records: list[dict[str, Any]] = []
    for speech_path in sorted(speech_dir.glob("*.wav")):
        item_id = speech_path.stem
        source_audio = audio_dir / f"{item_id}.wav"
        transcript_json = transcript_dir / f"{item_id}.json"
        metadata_json = metadata_dir / f"{item_id}.json"

        if not source_audio.exists():
            continue

        source_text = ""
        target_transcript = ""
        speaker_id = ""
        style = ""

        if metadata_json.exists():
            metadata = load_json(metadata_json)
            source_text = str(metadata.get("text", ""))
            speaker_id = str(metadata.get("speaker_id", ""))
            style = str(metadata.get("style", ""))

        if transcript_json.exists():
            transcript = load_json(transcript_json)
            target_transcript = str(transcript.get("transcript", ""))

        records.append(
            {
                "id": item_id,
                "speaker_id": speaker_id,
                "style": style,
                "source_text": source_text,
                "target_transcript": target_transcript,
                "source_audio": f"audio/{item_id}.wav",
                "target_speech": f"{target_speech_subdir}/{item_id}.wav",
            }
        )

        if max_samples is not None and len(records) >= max_samples:
            break

    return records


def write_html(export_dir: Path) -> None:
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Cascade Sample Viewer (Static)</title>
  <style>
    :root {{
      --bg: #f7f8fa;
      --fg: #0f172a;
      --muted: #475569;
      --card: #ffffff;
      --line: #e2e8f0;
      --accent: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 20px; background: var(--bg); color: var(--fg); font: 14px/1.4 -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; }}
    h1 {{ margin: 0 0 14px; }}
    .toolbar {{ display: grid; grid-template-columns: 1.5fr 1fr 1fr auto auto; gap: 10px; margin-bottom: 12px; }}
    input, select, button {{ width: 100%; padding: 8px 10px; border: 1px solid var(--line); border-radius: 8px; background: #fff; }}
    button {{ cursor: pointer; }}
    .meta {{ color: var(--muted); margin-bottom: 10px; }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 10px; background: #fff; margin-bottom: 14px; }}
    table {{ border-collapse: collapse; width: 100%; min-width: 980px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #f8fafc; z-index: 1; }}
    .cards {{ display: grid; gap: 12px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 12px; }}
    .card-head {{ font-weight: 600; margin-bottom: 8px; color: var(--accent); }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .label {{ color: var(--muted); margin-bottom: 4px; font-size: 12px; }}
    .text {{ background: #f8fafc; border: 1px solid var(--line); border-radius: 8px; padding: 8px; min-height: 56px; white-space: pre-wrap; }}
    audio {{ width: 100%; }}
    .pager {{ display: flex; gap: 8px; align-items: center; margin: 12px 0; }}
  </style>
</head>
<body>
  <h1>Cascade S2S Static Viewer</h1>
  <div class=\"meta\" id=\"summary\"></div>
  <div class=\"toolbar\">
    <input id=\"q\" placeholder=\"Search id/source text/target transcript\" />
    <select id=\"speaker\"><option value=\"\">All speakers</option></select>
    <select id=\"style\"><option value=\"\">All styles</option></select>
    <select id=\"pageSize\">
      <option>10</option><option selected>20</option><option>50</option><option>100</option>
    </select>
    <button id=\"reset\">Reset</button>
  </div>

  <div class=\"table-wrap\">
    <table>
      <thead><tr><th>ID</th><th>Speaker</th><th>Style</th><th>Source Text</th><th>Target Transcript</th></tr></thead>
      <tbody id=\"tbody\"></tbody>
    </table>
  </div>

  <div class=\"pager\">
    <button id=\"prev\">Prev</button>
    <span id=\"pageInfo\"></span>
    <button id=\"next\">Next</button>
  </div>

  <div class=\"cards\" id=\"cards\"></div>

<script src="./script.js"></script>
</body>
</html>
"""

    (export_dir / "index.html").write_text(html, encoding="utf-8")


def write_script(records: list[dict[str, Any]], export_dir: Path) -> None:
    data_json = json.dumps(records, ensure_ascii=False)
    script = f"""const DATA = {data_json};
const state = {{ q: "", speaker: "", style: "", page: 1, pageSize: 20 }};

const el = (id) => document.getElementById(id);
const esc = (s) => String(s || "")
  .replace(/&/g, "&amp;")
  .replace(/</g, "&lt;")
  .replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;")
  .replace(/'/g, "&#39;");

function fillFilters() {{
  const speakers = [...new Set(DATA.map(x => x.speaker_id).filter(Boolean))].sort();
  const styles = [...new Set(DATA.map(x => x.style).filter(Boolean))].sort();
  for (const s of speakers) el("speaker").insertAdjacentHTML("beforeend", `<option value="${{esc(s)}}">${{esc(s)}}</option>`);
  for (const s of styles) el("style").insertAdjacentHTML("beforeend", `<option value="${{esc(s)}}">${{esc(s)}}</option>`);
}}

function filtered() {{
  const q = state.q.trim().toLowerCase();
  return DATA.filter(x => {{
    if (state.speaker && x.speaker_id !== state.speaker) return false;
    if (state.style && x.style !== state.style) return false;
    if (!q) return true;
    return [x.id, x.source_text, x.target_transcript].join("\\n").toLowerCase().includes(q);
  }});
}}

function render() {{
  const arr = filtered();
  const total = arr.length;
  const totalPages = Math.max(1, Math.ceil(total / state.pageSize));
  if (state.page > totalPages) state.page = totalPages;
  const start = (state.page - 1) * state.pageSize;
  const end = Math.min(start + state.pageSize, total);
  const page = arr.slice(start, end);

  el("summary").textContent = `Total exported samples: ${{DATA.length}} | Matched: ${{total}}`;
  el("pageInfo").textContent = `Page ${{state.page}} / ${{totalPages}} (${{start + 1}}-${{end || 0}})`;

  el("tbody").innerHTML = page.map(x => `
    <tr>
      <td>${{esc(x.id)}}</td>
      <td>${{esc(x.speaker_id)}}</td>
      <td>${{esc(x.style)}}</td>
      <td>${{esc(x.source_text)}}</td>
      <td>${{esc(x.target_transcript)}}</td>
    </tr>
  `).join("");

  el("cards").innerHTML = page.map(x => `
    <section class="card">
      <div class="card-head">${{esc(x.id)}} | speaker=${{esc(x.speaker_id)}} | style=${{esc(x.style)}}</div>
      <div class="grid2">
        <div>
          <div class="label">Source Text</div>
          <div class="text">${{esc(x.source_text)}}</div>
          <div class="label" style="margin-top:8px;">Source Audio</div>
          <audio controls preload="none" src="${{esc(x.source_audio)}}"></audio>
        </div>
        <div>
          <div class="label">Target Transcript</div>
          <div class="text">${{esc(x.target_transcript)}}</div>
          <div class="label" style="margin-top:8px;">Target Speech</div>
          <audio controls preload="none" src="${{esc(x.target_speech)}}"></audio>
        </div>
      </div>
    </section>
  `).join("");
}}

el("q").addEventListener("input", (e) => {{ state.q = e.target.value; state.page = 1; render(); }});
el("speaker").addEventListener("change", (e) => {{ state.speaker = e.target.value; state.page = 1; render(); }});
el("style").addEventListener("change", (e) => {{ state.style = e.target.value; state.page = 1; render(); }});
el("pageSize").addEventListener("change", (e) => {{ state.pageSize = Number(e.target.value); state.page = 1; render(); }});
el("reset").addEventListener("click", () => {{
  state.q = ""; state.speaker = ""; state.style = ""; state.page = 1; state.pageSize = 20;
  el("q").value = ""; el("speaker").value = ""; el("style").value = ""; el("pageSize").value = "20";
  render();
}});
el("prev").addEventListener("click", () => {{ state.page = Math.max(1, state.page - 1); render(); }});
el("next").addEventListener("click", () => {{ state.page = state.page + 1; render(); }});

fillFilters();
render();
"""

    (export_dir / "script.js").write_text(script, encoding="utf-8")


def place_asset(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src)
    else:
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)


def export_assets(
    records: list[dict[str, Any]],
    outputs_dir: Path,
    export_dir: Path,
    target_speech_subdir: str,
    mode: str,
) -> None:
    dst_audio = export_dir / "audio"
    dst_speech = export_dir / target_speech_subdir
    dst_metadata = export_dir / "metadata"
    dst_transcript = export_dir / "transcript"
    dst_audio.mkdir(parents=True, exist_ok=True)
    dst_speech.mkdir(parents=True, exist_ok=True)
    dst_metadata.mkdir(parents=True, exist_ok=True)
    dst_transcript.mkdir(parents=True, exist_ok=True)

    for rec in records:
        item_id = rec["id"]
        place_asset(outputs_dir / "audio" / f"{item_id}.wav", dst_audio / f"{item_id}.wav", mode)
        place_asset(
            outputs_dir / target_speech_subdir / f"{item_id}.wav",
            dst_speech / f"{item_id}.wav",
            mode,
        )
        src_metadata = outputs_dir / "metadata" / f"{item_id}.json"
        src_transcript = outputs_dir / "transcript" / f"{item_id}.json"
        if src_metadata.exists():
            place_asset(src_metadata, dst_metadata / f"{item_id}.json", mode)
        if src_transcript.exists():
            place_asset(src_transcript, dst_transcript / f"{item_id}.json", mode)


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    export_dir = Path(args.export_dir)

    if not outputs_dir.exists():
        raise FileNotFoundError(f"outputs dir not found: {outputs_dir}")

    if export_dir.exists() and args.overwrite:
        shutil.rmtree(export_dir)

    export_dir.mkdir(parents=True, exist_ok=True)

    records = build_records(outputs_dir, args.target_speech_subdir, args.max_samples)
    if not records:
        raise RuntimeError(
            f"No eligible samples found (need matching {args.target_speech_subdir} + source audio)."
        )

    export_assets(records, outputs_dir, export_dir, args.target_speech_subdir, args.asset_mode)
    write_html(export_dir)
    write_script(records, export_dir)

    print(f"Export complete: {export_dir}")
    print(f"Samples: {len(records)}")
    print(f"Asset mode: {args.asset_mode}")
    print(f"Open: {export_dir / 'index.html'}")


if __name__ == "__main__":
    main()
