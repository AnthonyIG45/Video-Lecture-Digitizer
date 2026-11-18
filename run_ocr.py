import sys
import time

from block_ocr import BlockOCR


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else "frac_exponent_ex_3"
    blocks_dir = f"data/processed/blocks/{stem}"
    keyframes_dir = f"data/processed/keyframes/{stem}"

    start = time.time()
    BlockOCR().process_blocks(blocks_dir=blocks_dir, keyframes_dir=keyframes_dir)
    print(f"[*] Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
