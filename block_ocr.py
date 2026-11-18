import json
import os
import re
from pathlib import Path


class BlockOCR:
    """
    Runs Pix2Text on the block crops produced by BoardSegmenter. Each block is read as mixed
    text + LaTeX (printed prose becomes text, formulas become $$...$$).

    Figures are not detected: the formula recognizer silently ignores diagrams, so every
    block's crop is kept and linked next to its OCR text instead.
    """

    def __init__(self, device: str = "cpu"):
        # Imported here because pix2text is a heavy import and loads models on construction
        from pix2text import Pix2Text
        self.p2t = Pix2Text.from_config(enable_table=False, device=device)

    @staticmethod
    def _tidy(text: str) -> str:
        """Removes the spaces the recognizer inserts inside numbers: '1 9 . 0 8' -> '19.08'."""
        text = re.sub(r"(?<=\d) (?=[\d.])", "", text)
        return re.sub(r"(?<=\.) (?=\d)", "", text)

    def recognize(self, image_path: str) -> str:
        return self._tidy(self.p2t.recognize_text_formula(str(image_path), return_text=True).strip())

    @staticmethod
    def _timestamp(keyframe_name: str) -> str:
        m = re.match(r"keyframe_(\d+)m_(\d+)s", keyframe_name)
        return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}" if m else keyframe_name

    def process_blocks(self, blocks_dir: str, keyframes_dir: str = None) -> Path:
        """
        OCRs every block under blocks_dir (one sub-folder per keyframe), stores the result as 'text'
        in each blocks.json, and writes board_notes.md. Returns the path of the notes file.
        """
        root = Path(blocks_dir)
        notes_path = root / "board_notes.md"
        kf_dirs = sorted(d for d in root.iterdir() if d.is_dir() and (d / "blocks.json").exists())

        lines = [f"# Board notes: {root.name}\n"]
        for kf_dir in kf_dirs:
            blocks = json.loads((kf_dir / "blocks.json").read_text(encoding="utf-8"))
            lines.append(f"\n---\n\n## [{self._timestamp(kf_dir.name)}] {kf_dir.name}\n")

            if keyframes_dir:
                snapshot = Path(keyframes_dir) / f"{kf_dir.name}.jpg"
                if snapshot.exists():
                    rel = os.path.relpath(snapshot, root).replace(os.sep, "/")
                    lines.append(f"![{kf_dir.name}]({rel})\n")

            for i, block in enumerate(blocks, 1):
                crop = kf_dir / block["image"]
                try:
                    block["text"] = self.recognize(crop)
                except Exception as e:
                    block["text"] = ""
                    print(f"[-] OCR failed for {crop}: {e}")

                rel_crop = os.path.relpath(crop, root).replace(os.sep, "/")
                lines.append(f"\n**Block {i}**\n\n{block['text']}\n\n![block {i}]({rel_crop})\n")

            (kf_dir / "blocks.json").write_text(json.dumps(blocks, indent=2), encoding="utf-8")
            print(f"[+] {kf_dir.name}: {len(blocks)} blocks recognized")

        notes_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[+] Notes written to {notes_path.resolve()}")
        return notes_path
