import cv2
import numpy as np
from pathlib import Path

class BoardExtractor:
    """Handles automatic whiteboard/screen region-of-interest (ROI) detection and cropping."""

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.roi_bbox = None  # Stores (xmin, ymin, xmax, ymax) once computed

    def detect_board_automatically(self, sample_timestamps_sec: list = [2.0, 60.0]) -> tuple:
        """
        Scans multiple timestamps from the video (e.g., start and ~1 minute in) 
        to find the largest stable board rectangle, avoiding static intro cards.
        """
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path.resolve()}")

        cap = cv2.VideoCapture(str(video_path := str(self.video_path)))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 4.0  # Fallback to normalized default

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
                    # We want the largest valid quadrilateral (often the main projector/board)
                    if area > best_area:
                        best_area = area
                        x, y, bw, bh = cv2.boundingRect(approx)
                        best_box = (x, y, x + bw, y + bh)

        cap.release()

        if best_box is not None:
            self.roi_bbox = best_box
            print(f"[+] Board automatically detected (multi-sample scan): {self.roi_bbox}")
            return self.roi_bbox

        # Fallback to full frame bounds if no clear 4-point board contour is isolated
        print("[!] No distinct board rectangle found across samples. Defaulting to full frame ROI.")
        self.roi_bbox = (0, 0, w_frame, h_frame)
        return self.roi_bbox

    def crop_board(self, frame) -> np.ndarray:
        """
        Crops an incoming video frame to the established board bounding box.
        """
        if self.roi_bbox is None:
            self.detect_board_automatically()

        xmin, ymin, xmax, ymax = self.roi_bbox
        return frame[ymin:ymax, xmin:xmax]