from pathlib import Path
from video_processor import VideoProcessor


def main():
    raw_video = "data/raw/math15300_lecture.mp4"
    sliced_video = "data/processed/clips/clip_11_to_15.mp4"
    normalized_video = "data/processed/normalized/clip_11_to_15_norm.mp4"
    audio_file = "data/processed/audio/clip_11_to_15.wav"

    # Step 1: Slice the 4-minute segment (Instantaneous)
    print("[1/3] Slicing video...")
    VideoProcessor.slice_clip(
        input_path=raw_video,
        output_path=sliced_video,
        start_time="00:11:00",
        end_time="00:15:00"
    )

    # Step 2: Normalize down to 4 FPS (Drastically reduces vision workload)
    print("[2/3] Normalizing video to 4 FPS...")
    VideoProcessor.normalize_video(
        input_path=sliced_video,
        output_path=normalized_video,
        target_fps=4,
        max_width=1280
    )

    # Step 3: Extract dedicated Whisper-ready 16kHz WAV
    print("[3/3] Extracting audio for STT...")
    VideoProcessor.extract_audio_for_stt(
        input_path=sliced_video,
        output_path=audio_file
    )

    print("\n[+] Preprocessing Complete!")
    print(f"    - Normalized video: {Path(normalized_video).resolve()}")
    print(f"    - Clean WAV audio:  {Path(audio_file).resolve()}")


if __name__ == "__main__":
    main()