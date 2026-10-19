import cv2
import numpy as np
from pathlib import Path

class BoardExtractor:
    """Handles automatic whiteboard region-of-interest (ROI) detection and motion-settled keyframe extraction."""

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

    def extract_keyframes(self, output_dir: str, stability_duration_sec: float = 3.0, pixel_change_threshold: float = 1.5) -> list:
        """
        Scans the normalized video frame-by-frame, ignoring active writing motion,
        and saves a snapshot of the entire screen only when the board settles 
        for at least `stability_duration_sec`.
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
        prev_board_gray = None
        stable_frame_count = 0
        required_stable_frames = int(stability_duration_sec * fps)
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Crop to the established static whiteboard ROI
            board_roi = frame[ymin:ymax, xmin:xmax]
            gray_roi = cv2.cvtColor(board_roi, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)

            if prev_board_gray is not None:
                # 2. Measure pixel difference between current frame and previous frame
                diff = cv2.absdiff(prev_board_gray, gray_roi)
                mean_change = np.mean(diff)

                # If motion change is very low, the professor has stopped writing
                if mean_change < pixel_change_threshold:
                    stable_frame_count += 1
                else:
                    # Motion detected (writing, erasing, walking), reset timer
                    stable_frame_count = 0

                # 3. Trigger snapshot only when stability threshold is met
                if stable_frame_count == required_stable_frames:
                    timestamp_sec = frame_idx / fps
                    minutes = int(timestamp_sec // 60)
                    seconds = int(timestamp_sec % 60)
                    
                    filename = out_path / f"keyframe_{minutes:02d}m_{seconds:02d}s.jpg"
                    
                    # Save a snapshot of the full screen (or cropped board if preferred)
                    cv2.imwrite(str(filename), frame)
                    saved_snapshots.append(str(filename))
                    print(f"[+] Board settled! Captured keyframe at [{minutes:02d}:{seconds:02d}] -> {filename.name}")
                    
                    # Reset counter so it doesn't snapshot every frame while remaining idle
                    stable_frame_count = -int(10.0 * fps) # Cooldown guard: wait 10 seconds before next trigger

            prev_board_gray = gray_roi
            frame_idx += 1

        cap.release()
        print(f"[+] Feature Extraction Complete. Saved {len(saved_snapshots)} stable board keyframes.")
        return saved_snapshots