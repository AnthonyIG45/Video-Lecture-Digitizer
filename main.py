import os
from pathlib import Path

# Ensure ffmpeg environment is established before importing whisper
import imageio_ffmpeg
ffmpeg_dir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
if ffmpeg_dir not in os.environ["PATH"]:
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

from video_processor import VideoProcessor
from speech_to_text import LectureTranscriber


def run_pipeline(
    raw_video_path: str,
    target_fps: int = 4,
    whisper_model: str = "base",
    pause_threshold: float = 1.5,
):
    video_path = Path(raw_video_path)
    if not video_path.exists():
        print(f"[-] Video file not found: {video_path.resolve()}")
        return

    stem = video_path.stem
    print("=" * 60)
    print(f"[*] RUNNING PIPELINE: {video_path.name}")
    print("=" * 60)

    # Output paths
    norm_video_file = Path("data/processed/normalized") / f"{stem}_norm_{target_fps}fps.mp4"
    audio_file = Path("data/processed/audio") / f"{stem}_16k.wav"
    transcripts_dir = Path("data/processed/transcripts")

    # Step 1: Extract 16kHz audio from the entire video
    print("\n[1/3] Extracting 16kHz mono audio for Whisper...")
    VideoProcessor.extract_audio_for_stt(
        input_path=str(video_path),
        output_path=str(audio_file)
    )

    # Step 2: Transcribe with Whisper using pause segmentation
    print(f"\n[2/3] Transcribing with Whisper ('{whisper_model}') [Pause threshold: {pause_threshold}s]...")
    transcriber = LectureTranscriber(model_size=whisper_model)
    paragraphs = transcriber.transcribe(
        audio_or_video_path=str(audio_file),
        pause_threshold_sec=pause_threshold
    )

    txt_out, md_out = transcriber.export_transcripts(
        paragraphs=paragraphs,
        output_dir=transcripts_dir,
        file_stem=stem
    )

    # Step 3: Condense video to 4 FPS for later board/equation extraction
    print(f"\n[3/3] Normalizing video to {target_fps} FPS...")
    VideoProcessor.normalize_video(
        input_path=str(video_path),
        output_path=str(norm_video_file),
        target_fps=target_fps,
        max_width=1280
    )

    print("\n" + "=" * 60)
    print("[+] PIPELINE COMPLETE!")
    print(f"    - Clean Audio (WAV): {audio_file.resolve()}")
    print(f"    - Markdown Notes:    {md_out.resolve()}")
    print(f"    - Normalized Video:  {norm_video_file.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    # Point directly to your raw video file
    SOURCE_VIDEO = "data/raw/math15300_lecture.mp4"

    run_pipeline(
        raw_video_path=SOURCE_VIDEO,
        target_fps=4,
        whisper_model="base",
        pause_threshold=0.7
    )