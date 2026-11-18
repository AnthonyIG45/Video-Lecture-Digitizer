import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Block:
    bbox: tuple      # (x1, y1, x2, y2) in original image pixels
    ink_ratio: float


class BoardSegmenter:
    """
    Splits a board keyframe into line-level content blocks using classical OpenCV only:
    ink binarization followed by a recursive XY-cut that splits at whitespace gaps.
    Deciding what a block contains (text, formula or figure) is left to the OCR step.
    """

    def __init__(
        self,
        edge_margin_frac: float = 0.04,
        min_gap_y_frac: float = 0.012,
        min_gap_x_frac: float = 0.07,
        ink_threshold: int = 170,
        min_block_area_frac: float = 0.0006,
        min_edge_block_height_frac: float = 0.06,
        pad_px: int = 8,
        ignore_boxes: list = None,
    ):
        self.edge_margin_frac = edge_margin_frac   # ignore ink this close to the edge (UI chrome, borders)
        self.min_gap_y_frac = min_gap_y_frac       # blank rows (fraction of height) that split two lines
        self.min_gap_x_frac = min_gap_x_frac       # blank columns (fraction of width) that split two blocks on a line
        self.ink_threshold = ink_threshold         # pixel counts as ink if darker than this (0-255) vs. local background
        self.min_block_area_frac = min_block_area_frac
        self.min_edge_block_height_frac = min_edge_block_height_frac  # blocks touching the edge margin below this are clipped fragments
        self.pad_px = pad_px
        # Regions to blank out, as (x1, y1, x2, y2) fractions of the content area, e.g. app toolbars
        self.ignore_boxes = ignore_boxes or []

    @staticmethod
    def _content_area(gray: np.ndarray) -> tuple:
        """Bounds of the non-black region, to skip letterbox bars. Returns (x1, y1, x2, y2)."""
        rows = np.where(gray.mean(axis=1) > 8)[0]
        cols = np.where(gray.mean(axis=0) > 8)[0]
        if len(rows) == 0 or len(cols) == 0:
            return 0, 0, gray.shape[1], gray.shape[0]
        return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1

    def _ink_mask(self, gray: np.ndarray) -> np.ndarray:
        """Binary mask (255 = ink) that is robust to uneven lighting and to chalk-on-dark boards."""
        if np.median(gray) < 110:
            gray = 255 - gray

        # Divide by a heavily blurred copy to flatten the background to ~255
        sigma = max(gray.shape) / 30
        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
        flat = cv2.divide(gray, background, scale=255)
        return np.where(flat < self.ink_threshold, 255, 0).astype(np.uint8)

    def _clean_ink(self, ink: np.ndarray) -> np.ndarray:
        """Drops specks, edge ink and long thin lines (scrollbars, dividers)."""
        h, w = ink.shape
        mx, my = int(w * self.edge_margin_frac), int(h * self.edge_margin_frac)
        ink[:my, :] = 0
        ink[h - my:, :] = 0
        ink[:, :mx] = 0
        ink[:, w - mx:] = 0
        for fx1, fy1, fx2, fy2 in self.ignore_boxes:
            ink[int(fy1 * h):int(fy2 * h), int(fx1 * w):int(fx2 * w)] = 0

        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        min_area = max(8, int(h * w * 0.00001))

        drop = np.zeros(n, bool)
        for i in range(1, n):
            _, _, cw, ch, area = stats[i]
            is_long_vline = ch > 0.6 * h and cw < 0.01 * w
            is_long_hline = cw > 0.6 * w and ch < 0.01 * h
            drop[i] = area < min_area or is_long_vline or is_long_hline
        ink[drop[labels]] = 0
        return ink

    @staticmethod
    def _largest_gap(profile: np.ndarray, min_gap: int):
        """Midpoint of the longest run of empty bins strictly inside the ink extent, or None if shorter than min_gap."""
        filled = np.where(profile > 0)[0]
        if len(filled) == 0:
            return None
        start, end = filled[0], filled[-1]

        best_len, best_mid, run_start = 0, None, None
        for i in range(start, end + 1):
            if profile[i] == 0:
                if run_start is None:
                    run_start = i
            elif run_start is not None:
                if i - run_start > best_len:
                    best_len, best_mid = i - run_start, (run_start + i) // 2
                run_start = None
        return best_mid if best_len >= min_gap else None

    def _xy_cut(self, ink: np.ndarray, x1: int, y1: int, x2: int, y2: int, min_gap_x: int, min_gap_y: int) -> list:
        """Recursively splits a region at its widest whitespace gap (rows first, then columns). Returns leaf boxes."""
        region = ink[y1:y2, x1:x2]

        cut = self._largest_gap(region.sum(axis=1), min_gap_y)
        if cut is not None:
            return (
                self._xy_cut(ink, x1, y1, x2, y1 + cut, min_gap_x, min_gap_y)
                + self._xy_cut(ink, x1, y1 + cut, x2, y2, min_gap_x, min_gap_y)
            )

        cut = self._largest_gap(region.sum(axis=0), min_gap_x)
        if cut is not None:
            return (
                self._xy_cut(ink, x1, y1, x1 + cut, y2, min_gap_x, min_gap_y)
                + self._xy_cut(ink, x1 + cut, y1, x2, y2, min_gap_x, min_gap_y)
            )

        ys, xs = np.where(region > 0)
        if len(xs) == 0:
            return []
        return [(x1 + int(xs.min()), y1 + int(ys.min()), x1 + int(xs.max()) + 1, y1 + int(ys.max()) + 1)]

    @staticmethod
    def _merge_stacked(boxes: list, board_w: int, board_h: int, max_width_frac: float = 0.4, max_gap_frac: float = 0.04) -> list:
        """
        Re-joins fractions that the row cut split: a narrow box with a box directly below it that sits
        within its horizontal extent (numerator over denominator). Wide text lines are left alone.
        """
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        merged = True
        while merged:
            merged = False
            for i, a in enumerate(boxes):
                if a[2] - a[0] > max_width_frac * board_w:
                    continue
                slack = 0.15 * (a[2] - a[0])
                for j, b in enumerate(boxes):
                    if i == j or b[1] < a[1]:
                        continue
                    gap = b[1] - a[3]
                    inside = b[0] >= a[0] - slack and b[2] <= a[2] + slack
                    if gap < max_gap_frac * board_h and inside:
                        boxes[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                        boxes.pop(j)
                        merged = True
                        break
                if merged:
                    break
        return boxes

    def _keep(self, box: tuple, ink: np.ndarray) -> bool:
        """Filters out specks and fragments that were clipped by the edge margin."""
        h, w = ink.shape
        x1, y1, x2, y2 = box
        if (x2 - x1) * (y2 - y1) < self.min_block_area_frac * w * h:
            return False

        mx, my = int(w * self.edge_margin_frac), int(h * self.edge_margin_frac)
        touches_edge = y1 <= my + 2 or y2 >= h - my - 2 or x1 <= mx + 2 or x2 >= w - mx - 2
        if touches_edge and (y2 - y1) < self.min_edge_block_height_frac * h:
            return False
        return True

    def segment(self, image_path: str) -> tuple:
        """Returns (blocks, image) with blocks in reading order and bboxes in original image pixels."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ox1, oy1, ox2, oy2 = self._content_area(gray_full)
        gray = gray_full[oy1:oy2, ox1:ox2]
        h, w = gray.shape

        ink = self._clean_ink(self._ink_mask(gray))
        min_gap_x = max(3, int(w * self.min_gap_x_frac))
        min_gap_y = max(3, int(h * self.min_gap_y_frac))

        blocks = []
        boxes = self._merge_stacked(self._xy_cut(ink, 0, 0, w, h, min_gap_x, min_gap_y), w, h)
        for box in boxes:
            if not self._keep(box, ink):
                continue
            tx1, ty1, tx2, ty2 = box
            ink_ratio = float(np.count_nonzero(ink[ty1:ty2, tx1:tx2])) / max((tx2 - tx1) * (ty2 - ty1), 1)
            bbox = (
                max(0, tx1 - self.pad_px) + ox1,
                max(0, ty1 - self.pad_px) + oy1,
                min(w, tx2 + self.pad_px) + ox1,
                min(h, ty2 + self.pad_px) + oy1,
            )
            blocks.append(Block(bbox, round(ink_ratio, 3)))

        blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
        return blocks, img

    def process_keyframes(self, keyframes_dir: str, output_dir: str) -> dict:
        """Segments every keyframe; saves block crops, an annotated preview and a JSON file per keyframe."""
        out_root = Path(output_dir)
        results = {}

        keyframes = sorted(Path(keyframes_dir).glob("keyframe_*m_*s.jpg"))
        for kf in keyframes:
            blocks, img = self.segment(str(kf))
            kf_dir = out_root / kf.stem
            kf_dir.mkdir(parents=True, exist_ok=True)

            annotated = img.copy()
            records = []
            for i, b in enumerate(blocks, 1):
                x1, y1, x2, y2 = b.bbox
                crop_name = f"block{i:02d}.png"
                cv2.imwrite(str(kf_dir / crop_name), img[y1:y2, x1:x2])

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 160, 0), 2)
                cv2.putText(annotated, str(i), (x1 + 4, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 160, 0), 2)
                records.append({**asdict(b), "image": crop_name})

            cv2.imwrite(str(kf_dir / "_annotated.jpg"), annotated)
            (kf_dir / "blocks.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
            results[kf.name] = records
            print(f"[+] {kf.name}: {len(blocks)} blocks")
        return results
