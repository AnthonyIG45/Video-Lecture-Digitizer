from pathlib import Path
from speech_to_text import LectureTranscriber


def main():
    # Target media: can be the clean .wav from VideoProcessor or the sliced .mp4
    input_audio = "data/processed/audio/clip_11_to_15.wav"
    output_dir = "data/processed/transcripts"

    if not Path(input_audio).exists():
        print(f"[-] Target file {input_audio} does not exist. Run run_preprocess.py first.")
        return

    # Use 'base' for quick turnaround; use 'small' or 'medium' for denser math lecture terminology
    transcriber = LectureTranscriber(model_size="base")

    paragraphs = transcriber.transcribe(
        audio_or_video_path=input_audio,
        pause_threshold_sec=1.5
    )

    print(f"[*] Processed {len(paragraphs)} paragraph blocks.")
    transcriber.export_transcripts(
        paragraphs=paragraphs,
        output_dir=output_dir,
        file_stem="clip_11_to_15"
    )


if __name__ == "__main__":
    main()