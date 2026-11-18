import cv2
import numpy as np


def flatten_background(gray: np.ndarray) -> np.ndarray:
    """
    Grayscale board image with the background flattened to ~255, so ink and shadows can be
    thresholded relative to local paper brightness. Handles chalk-on-dark boards by inverting them.
    """
    if np.median(gray) < 110:
        gray = 255 - gray

    # Divide by a heavily blurred copy to flatten the background
    sigma = max(gray.shape) / 30
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    return cv2.divide(gray, background, scale=255)


def ink_mask(gray: np.ndarray, threshold: int = 170) -> np.ndarray:
    """
    Binary mask (255 = ink), robust to uneven lighting, exposure flicker and chalk-on-dark boards.
    A pixel counts as ink if it is darker than `threshold` (0-255) relative to its local background.
    """
    return np.where(flatten_background(gray) < threshold, 255, 0).astype(np.uint8)
