import sys
import time

from board_segmenter import BoardSegmenter

# Per-video regions to blank out (fractions of the frame), e.g. app banners in screen recordings
IGNORE_BOXES = {
    # OneNote screen recording: banner/toolbar strip at the top
    "frac_exponent_ex_3": [(0.0, 0.0, 1.0, 0.125)],
}


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else "frac_exponent_ex_3"
    keyframes_dir = f"data/processed/keyframes/{stem}"
    output_dir = f"data/processed/blocks/{stem}"

    start = time.time()
    segmenter = BoardSegmenter(ignore_boxes=IGNORE_BOXES.get(stem, []))
    segmenter.process_keyframes(keyframes_dir=keyframes_dir, output_dir=output_dir)
    print(f"[*] Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
