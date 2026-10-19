import cv2
import numpy as np
from pathlib import Path

class BoardExtractor:
    """Handles automatic whiteboard region-of-interest (ROI) detection, motion-settled keyframe extraction, and SSIM deduplication."""

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
    def _calculate_ssim(img1, img2) -> float:
        """
        Computes normalized cross-correlation similarity between two frames (0.0 to 1.0).
        1.0 means identical, 0.0 means completely unrelated.
        """
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Resize to a standard low-res grid for fast, robust comparison
        g1 = cv2.resize(g1, (256, 256))
        g2 = cv2.resize(g2, (256, 256))

        # Use normalized cross-correlation
        res = cv2.matchTemplate(g1, g2, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        
        # max_val ranges from -1.0 to 1.0; clamp to 0.0 to 1.0
        return float(max(0.0, max_val))

    def extract_keyframes(
        self, 
        output_dir: str, 
        stability_duration_sec: float = 3.0, 
        pixel_change_threshold: float = 1.5,
        ssim_redundancy_threshold: float = 0.95
    ) -> list:
        """
        Scans video for stable board states, filtering out redundant snapshots 
        via structural similarity (SSIM) comparison against the last saved frame.
        """
        if self.roi_bbox is None:
            self.detect_board_automatically()

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 4.0

        xmin, ymin, xmax, ymax = self.roi_bbox
        
        saved_snapshots = []
        last_saved_frame = None
        prev_board_gray = None
        stable_frame_count = 0
        required_stable_frames = int(stability_duration_sec * fps)
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            board_roi = frame[ymin:ymax, xmin:xmax]
            gray_roi = cv2.cvtColor(board_roi, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)

            if prev_board_gray is not None:
                diff = cv2.absdiff(prev_board_gray, gray_roi)
                mean_change = np.mean(diff)

                if mean_change < pixel_change_threshold:
                    stable_frame_count += 1
                else:
                    stable_frame_count = 0

                if stable_frame_count == required_stable_frames:
                    # Check SSIM redundancy against the last successfully saved keyframe
                    is_redundant = False
                    if last_saved_frame is not None:
                        similarity = self._calculate_ssim(last_saved_frame, frame)
                        if similarity >= ssim_redundancy_threshold:
                            is_redundant = True

                    if not is_redundant:
                        timestamp_sec = frame_idx / fps
                        minutes = int(timestamp_sec // 60)
                        seconds = int(timestamp_sec % 60)
                        
                        filename = out_path / f"keyframe_{minutes:02d}m_{seconds:02d}s.jpg"
                        cv2.imwrite(str(filename), frame)
                        saved_snapshots.append(str(filename))
                        last_saved_frame = frame.copy()
                        print(f"[+] Unique board state captured at [{minutes:02d}:{seconds:02d}] -> {filename.name}")
                    else:
                        print(f"[-] Skipping redundant board state at frame {frame_idx}")

                    stable_frame_count = -int(10.0 * fps) # 10-second cooldown guard

            prev_board_gray = gray_roi
            frame_idx += 1

        cap.release()
        print(f"[+] Deduplication Complete. Retained {len(saved_snapshots)} distinct keyframes.")
        return saved_snapshots