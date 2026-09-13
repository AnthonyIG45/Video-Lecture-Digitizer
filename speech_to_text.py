import os
import shutil
from pathlib import Path
import imageio_ffmpeg

# Ensure ffmpeg binary is resolved
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = str(Path(ffmpeg_exe).parent)
target_alias = Path(ffmpeg_dir) / "ffmpeg.exe"

if not target_alias.exists():
    shutil.copyfile(ffmpeg_exe, target_alias)

if ffmpeg_dir not in os.environ["PATH"]:
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

import whisper


class LectureTranscriber:
    def __init__(self, model_size: str = "base", device: str = None):
        print(f"[*] Initializing Whisper model: '{model_size}'...")
        self.model = whisper.load_model(model_size, device=device)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def transcribe(
        self,
        audio_or_video_path: str,
        pause_threshold_sec: float = 1.5,
        language: str = "en"
    ):
        media_path = Path(audio_or_video_path)
        if not media_path.exists():
            raise FileNotFoundError(f"Audio file not found: {media_path.resolve()}")

        print(f"[*] Transcribing: {media_path.name} (word timestamps enabled)...")
        
        # word_timestamps=True reveals the exact timing of every single word
        result = self.model.transcribe(
            str(media_path),
            language=language,
            word_timestamps=True,
            verbose=False
        )

        segments = result.get("segments", [])
        if not segments:
            return []

        # Flatten out all words with their timings
        all_words = []
        for segment in segments:
            for w in segment.get("words", []):
                all_words.append({
                    "word": w["word"].strip(),
                    "start": w["start"],
                    "end": w["end"]
                })

        if not all_words:
            # Fallback if no words dictionary was generated
            return [{"start": segments[0]["start"], "end": segments[-1]["end"], "text": result.get("text", "")}]

        paragraphs = []
        current_words = []
        para_start = all_words[0]["start"]
        last_word_end = all_words[0]["end"]

        for item in all_words:
            word = item["word"]
            start = item["start"]
            end = item["end"]

            # Measure silence gap between current word start and previous word end
            silence_gap = start - last_word_end

            if silence_gap >= pause_threshold_sec and current_words:
                paragraphs.append({
                    "start": para_start,
                    "end": last_word_end,
                    "text": " ".join(current_words)
                })
                current_words = [word]
                para_start = start
            else:
                current_words.append(word)

            last_word_end = end

        if current_words:
            paragraphs.append({
                "start": para_start,
                "end": last_word_end,
                "text": " ".join(current_words)
            })

        return paragraphs

    def export_transcripts(self, paragraphs, output_dir, file_stem: str):
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        txt_file = out_dir / f"{file_stem}_transcript.txt"
        md_file = out_dir / f"{file_stem}_transcript.md"

        # Plain text: separated by blank lines (white space)
        with open(txt_file, "w", encoding="utf-8") as f:
            for p in paragraphs:
                f.write(p["text"] + "\n\n")

        # Markdown: readable timestamped paragraphs with visual spacing
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(f"# Lecture Transcript: {file_stem}\n\n---\n\n")
            for p in paragraphs:
                ts = f"**[{self._format_timestamp(p['start'])} - {self._format_timestamp(p['end'])}]**"
                f.write(f"{ts}\n\n{p['text']}\n\n")

        print(f"[+] Transcripts saved successfully:")
        print(f"    - Plaintext: {txt_file.resolve()}")
        print(f"    - Markdown:  {md_file.resolve()}")

        return txt_file, md_file