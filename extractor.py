import cv2
import numpy as np
from collections import deque
from pathlib import Path

from ink import flatten_background


class BoardExtractor:
    """Handles automatic whiteboard region-of-interest (ROI) detection and ink-based keyframe extraction."""

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.roi_bbox = None

    def detect_board_automatically(self, sample_timestamps_sec: list = [2.0, 60.0]) -> tuple:
        """
        Automatically detects the whiteboard/projection screen ROI.
        """
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path.resolve()}")

        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 4.0

        best_area = 0
        best_box = None
        h_frame, w_frame = 720, 1280

        for t_sec in sample_timestamps_sec:
            target_frame_idx = int(t_sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            ret, frame = cap.read()

            if not ret or frame is None:
                continue

            h_frame, w_frame = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 50, 150)

            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            min_area = h_frame * w_frame * 0.15

            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                if len(approx) == 4 and cv2.contourArea(c) > min_area:
                    area = cv2.contourArea(c)
                    if area > best_area:
                        best_area = area
                        x, y, bw, bh = cv2.boundingRect(approx)
                        best_box = (x, y, x + bw, y + bh)

        cap.release()

        if best_box is not None:
            self.roi_bbox = best_box
            return self.roi_bbox

        self.roi_bbox = (0, 0, w_frame, h_frame)
        return self.roi_bbox

    @staticmethod
    def _analyze(board_roi: np.ndarray, width: int = 480) -> tuple:
        """
        Cheap per-frame analysis at reduced resolution. Returns (ink mask, occlusion fraction), where
        occlusion is the share of the board covered by large dark blobs (a hand or pen): thin
        handwriting strokes vanish under a morphological opening, big blobs do not.
        """
        h, w = board_roi.shape[:2]
        scale = min(1.0, width / w)
        small = cv2.resize(board_roi, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        flat = flatten_background(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))

        ink = np.where(flat < 170, 255, 0).astype(np.uint8)
        shadow = np.where(flat < 215, 255, 0).astype(np.uint8)
        blobs = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        return ink, float(np.count_nonzero(blobs)) / blobs.size

    @staticmethod
    def _ink_change(a: np.ndarray, b: np.ndarray, kernel: np.ndarray) -> tuple:
        """(added, removed) ink pixel counts going from mask a to mask b, ignoring a few pixels of jitter."""
        added = np.count_nonzero(cv2.subtract(b, cv2.dilate(a, kernel)))
        removed = np.count_nonzero(cv2.subtract(a, cv2.dilate(b, kernel)))
        return added, removed

    def extract_keyframes(
        self,
        output_dir: str,
        stability_duration_sec: float = 3.0,
        ink_change_threshold: float = 0.002,
        min_changed_ink_fraction: float = 0.10,
        cooldown_sec: float = 4.0,
        max_wait_sec: float = 20.0,
        occlusion_window_sec: float = 4.0,
        max_clean_occlusion: float = 0.006,
        extra_wait_sec: float = 12.0,
    ) -> list:
        """
        Saves a snapshot of the board each time its writing reaches a new state. Two triggers:

        1. Settled: the ink has stayed essentially unchanged for `stability_duration_sec` (changed
           pixels below `ink_change_threshold` of the frame, measured against the start of the window
           so slow handwriting still counts as change). Working on ink rather than raw pixels makes
           this immune to lighting flicker, camera noise and shadows.
        2. Progress: no snapshot for `max_wait_sec`. Lecturers who write continuously with a hand in
           frame (e.g. a webcam over paper) never settle, so the least-occluded frame of the last
           `occlusion_window_sec` is taken instead. If even that frame has a hand over the board
           (occlusion above `max_clean_occlusion`), it waits up to `extra_wait_sec` more for a clear one.

        Either way a snapshot is kept only if at least `min_changed_ink_fraction` of its ink differs
        from the last saved one, and blank boards are skipped.
        """
        if self.roi_bbox is None:
            self.detect_board_automatically()

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 4.0

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Clamp ROI to the frame; fall back to the full frame if it ends up empty
        xmin, ymin, xmax, ymax = self.roi_bbox
        xmin, xmax = max(0, xmin), min(frame_w, xmax)
        ymin, ymax = max(0, ymin), min(frame_h, ymax)
        if xmax <= xmin or ymax <= ymin:
            xmin, ymin, xmax, ymax = 0, 0, frame_w, frame_h

        kernel = np.ones((5, 5), np.uint8)
        required_stable_frames = int(stability_duration_sec * fps)
        cooldown_frames = int(cooldown_sec * fps)
        max_wait_frames = int(max_wait_sec * fps)
        extra_wait_frames = int(extra_wait_sec * fps)

        saved_snapshots = []
        last_saved_ink = None
        anchor_ink = None          # ink at the start of the current candidate-stable window
        stable_frames = 0
        evaluated = False          # this settled window has already been judged (saved or skipped)
        cooldown_left = 0
        next_progress_check = max_wait_frames
        progress_due_since = None  # frame at which a progress snapshot first became due
        recent = deque(maxlen=max(1, int(occlusion_window_sec * fps)))  # (idx, board_roi, ink, occlusion)

        def keep_if_new(idx: int, roi: np.ndarray, ink: np.ndarray) -> bool:
            """Saves the snapshot if it shows enough new/changed writing. Returns True if saved."""
            nonlocal last_saved_ink, cooldown_left
            ink_px = np.count_nonzero(ink)
            if ink_px < 0.0005 * ink.size:
                return False  # blank board: nothing written yet

            if last_saved_ink is None:
                changed = 1.0
            else:
                added, removed = self._ink_change(last_saved_ink, ink, kernel)
                changed = (added + removed) / max(np.count_nonzero(last_saved_ink), ink_px)

            timestamp_sec = idx / fps
            minutes, seconds = int(timestamp_sec // 60), int(timestamp_sec % 60)
            if changed < min_changed_ink_fraction:
                print(f"[-] Skipping unchanged board state at [{minutes:02d}:{seconds:02d}]")
                return False

            filename = out_path / f"keyframe_{minutes:02d}m_{seconds:02d}s.jpg"
            if str(filename) in saved_snapshots:
                return False  # same second as an existing snapshot
            cv2.imwrite(str(filename), roi, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_snapshots.append(str(filename))
            last_saved_ink = ink
            cooldown_left = cooldown_frames
            print(f"[+] New board state captured at [{minutes:02d}:{seconds:02d}] -> {filename.name}")
            return True

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            board_roi = frame[ymin:ymax, xmin:xmax]
            ink, occlusion = self._analyze(board_roi)
            recent.append((frame_idx, board_roi.copy(), ink, occlusion))
            cooldown_left = max(0, cooldown_left - 1)

            if anchor_ink is None:
                anchor_ink = ink
            else:
                added, removed = self._ink_change(anchor_ink, ink, kernel)
                if (added + removed) / ink.size < ink_change_threshold:
                    stable_frames += 1
                else:
                    anchor_ink, stable_frames, evaluated = ink, 0, False

            if stable_frames >= required_stable_frames and not evaluated and cooldown_left == 0:
                evaluated = True
                if keep_if_new(frame_idx, board_roi, ink):
                    next_progress_check = frame_idx + max_wait_frames
                    progress_due_since = None

            elif frame_idx >= next_progress_check and cooldown_left == 0:
                # Writing never settled: take the clearest recent frame (least hand/pen in view),
                # waiting a little for a clear moment if a hand is still over the board
                if progress_due_since is None:
                    progress_due_since = frame_idx
                best_idx, best_roi, best_ink, best_occlusion = min(recent, key=lambda r: (r[3], -r[0]))
                if best_occlusion <= max_clean_occlusion or frame_idx - progress_due_since >= extra_wait_frames:
                    keep_if_new(best_idx, best_roi, best_ink)
                    next_progress_check = frame_idx + max_wait_frames
                    progress_due_since = None

            frame_idx += 1

        cap.release()
        print(f"[+] Extraction complete. Retained {len(saved_snapshots)} distinct keyframes.")
        return saved_snapshots
