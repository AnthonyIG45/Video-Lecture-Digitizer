"""
Storyboard data contract: the single hand-off between the processing pipeline and anything that
presents its results (PDF builder, UI). This module only needs the standard library.

storyboard.json layout (all image paths are relative to the JSON file's folder):
{
  "version": 1,
  "title": "Lecture title",
  "source": "where the video came from",
  "sections": [
    {
      "start": 0.0, "end": 21.3, "speaker": "Professor", "text": "transcript paragraph",
      "snapshots": [
        {"time": 3, "image": "../keyframes/x/keyframe_00m_03s.jpg",
         "blocks": [{"text": "OCR text/LaTeX", "image": "../blocks/x/keyframe_00m_03s/block01.png"}]}
      ]
    }
  ]
}
"""
import json
import os
from pathlib import Path

STORYBOARD_VERSION = 1


def index_keyframes(keyframes_dir) -> dict:
    """Maps timestamp in seconds -> keyframe path, parsed from names like 'keyframe_02m_43s.jpg'."""
    keyframe_map = {}
    if keyframes_dir and Path(keyframes_dir).exists():
        for kf_path in Path(keyframes_dir).glob("keyframe_*m_*s.jpg"):
            try:
                parts = kf_path.stem.replace("keyframe_", "").split("m_")
                keyframe_map[int(parts[0]) * 60 + int(parts[1].replace("s", ""))] = kf_path.resolve()
            except (IndexError, ValueError):
                continue
    return keyframe_map


def place_keyframes(paragraphs: list, keyframe_map: dict) -> dict:
    """
    Assigns every keyframe to the latest paragraph that started at or before it (the first paragraph
    if it precedes all speech), so no snapshot is dropped. Returns {paragraph index: [(seconds, path)]}.
    """
    placed = {}
    for kf_sec, kf_path in sorted(keyframe_map.items()):
        idx = max((i for i, p in enumerate(paragraphs) if p["start"] <= kf_sec), default=0)
        placed.setdefault(idx, []).append((kf_sec, kf_path))
    return placed


def write_storyboard(
    paragraphs: list,
    keyframes_dir,
    blocks_dir,
    output_path,
    title: str,
    source: str = None,
) -> Path:
    """Writes storyboard.json and returns its path."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def rel(path) -> str:
        return os.path.relpath(path, out.parent.resolve()).replace(os.sep, "/")

    keyframe_map = index_keyframes(keyframes_dir)
    placed = place_keyframes(paragraphs, keyframe_map)

    # Snapshots with no transcript at all still need a section to live in
    if not paragraphs and keyframe_map:
        paragraphs = [{"start": 0.0, "end": 0.0, "speaker": "", "text": ""}]
        placed = place_keyframes(paragraphs, keyframe_map)

    sections = []
    for i, p in enumerate(paragraphs):
        snapshots = []
        for kf_sec, kf_path in placed.get(i, []):
            blocks = []
            blocks_json = Path(blocks_dir) / kf_path.stem / "blocks.json" if blocks_dir else None
            if blocks_json and blocks_json.exists():
                for b in json.loads(blocks_json.read_text(encoding="utf-8")):
                    blocks.append({"text": b.get("text", ""), "image": rel(blocks_json.parent.resolve() / b["image"])})
            snapshots.append({"time": kf_sec, "image": rel(kf_path), "blocks": blocks})

        sections.append({
            "start": float(p["start"]),
            "end": float(p["end"]),
            "speaker": p["speaker"],
            "text": p["text"],
            "snapshots": snapshots,
        })

    data = {"version": STORYBOARD_VERSION, "title": title, "source": source, "sections": sections}
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
