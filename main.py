import os
from pathlib import Path

# Ensure ffmpeg environment is established before importing whisper
import imageio_ffmpeg
ffmpeg_dir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
if ffmpeg_dir not in os.environ["PATH"]:
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

from video_processor import VideoProcessor
from speech_to_text import LectureTranscriber
from extractor import BoardExtractor
from board_segmenter import BoardSegmenter
from block_ocr import BlockOCR
from storyboard import write_storyboard


def run_pipeline(
    raw_video_path: str,
    target_fps: int = 4,
    whisper_model: str = "base",
    pause_threshold: float = 1.5,
    ignore_boxes: list = None,
    title: str = None,
    source: str = None,
    progress=print,
    work_dir="data",
) -> Path:
    """
    Runs the full pipeline on a local video and returns the path of the storyboard.json it produced
    (see storyboard.py for the layout). `progress` receives one status line per pipeline step.
    Every intermediate file is written under `<work_dir>/processed`, so pointing work_dir at a
    temporary folder keeps the run from leaving anything behind.
    """
    video_path = Path(raw_video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path.resolve()}")

    stem = video_path.stem
    print("=" * 60)
    print(f"[*] RUNNING PIPELINE: {video_path.name}")
    print("=" * 60)

    # Output paths
    processed = Path(work_dir) / "processed"
    norm_video_file = processed / "normalized" / f"{stem}_norm_{target_fps}fps.mp4"
    audio_file = processed / "audio" / f"{stem}_16k.wav"
    transcripts_dir = processed / "transcripts"
    keyframe_dir = processed / "keyframes" / stem
    blocks_dir = processed / "blocks" / stem
    storyboard_file = processed / "storyboards" / f"{stem}.json"

    # Step 1: Extract 16kHz audio
    progress("[1/5] Extracting 16kHz mono audio for Whisper...")
    VideoProcessor.extract_audio_for_stt(
        input_path=str(video_path),
        output_path=str(audio_file)
    )

    # Step 2: Condense video to 4 FPS & run Feature Extraction (Board ROI + SSIM Keyframes)
    progress("[2/5] Normalizing video & extracting motion-settled keyframes...")
    VideoProcessor.normalize_video(
        input_path=str(video_path),
        output_path=str(norm_video_file),
        target_fps=target_fps,
        max_width=1280
    )

    extractor = BoardExtractor(video_path=str(norm_video_file))
    board_bounds = extractor.detect_board_automatically(sample_timestamps_sec=[2.0, 60.0])
    print(f"    - Active Board ROI: {board_bounds}")
    
    saved_keyframes = extractor.extract_keyframes(
        output_dir=str(keyframe_dir),
        stability_duration_sec=3.0,
        pixel_change_threshold=1.5,
        ssim_redundancy_threshold=0.95
    )

    # Step 3: Split each keyframe into line blocks and OCR them (Pix2Text)
    progress("[3/5] Segmenting board blocks & running OCR...")
    BoardSegmenter(ignore_boxes=ignore_boxes).process_keyframes(
        keyframes_dir=str(keyframe_dir),
        output_dir=str(blocks_dir)
    )
    BlockOCR().process_blocks(
        blocks_dir=str(blocks_dir),
        keyframes_dir=str(keyframe_dir)
    )

    # Step 4: Transcribe with Whisper (with speaker diarization & pause segmentation)
    progress(f"[4/5] Transcribing with Whisper ('{whisper_model}')...")
    transcriber = LectureTranscriber(model_size=whisper_model)
    paragraphs = transcriber.transcribe(
        audio_or_video_path=str(audio_file),
        pause_threshold_sec=pause_threshold
    )

    # Step 5: Export Unified Storyboard (Markdown mapping transcript + keyframes + board OCR)
    progress("[5/5] Generating unified multimedia storyboard...")
    transcriber.export_transcripts(
        paragraphs=paragraphs,
        output_dir=transcripts_dir,
        file_stem=stem,
        keyframes_dir=str(keyframe_dir),
        blocks_dir=str(blocks_dir)
    )
    return write_storyboard(
        paragraphs=paragraphs,
        keyframes_dir=keyframe_dir,
        blocks_dir=blocks_dir,
        output_path=storyboard_file,
        title=title or stem.replace("_", " ").title(),
        source=source or video_path.name,
    )


if __name__ == "__main__":
    # Point directly to your raw video file
    """
    SOURCE_VIDEO = "data/raw/math15300_lecture.mp4"

    run_pipeline(
        raw_video_path=SOURCE_VIDEO,
        target_fps=4,
        whisper_model="base",
        pause_threshold=0.7
    )
    """

    SOURCE_VIDEO = "data/raw/frac_exponent_ex_3.mp4"

    run_pipeline(
        raw_video_path=SOURCE_VIDEO,
        target_fps=4,
        whisper_model="base",
        pause_threshold=0.7,
        # OneNote screen recording: blank out the app banner/toolbar strip at the top
        ignore_boxes=[(0.0, 0.0, 1.0, 0.125)]
    )