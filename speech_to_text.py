import os
import shutil
import wave
from pathlib import Path
import numpy as np
import imageio_ffmpeg

# Ensure ffmpeg binary is resolved for Whisper
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = str(Path(ffmpeg_exe).parent)
target_alias = Path(ffmpeg_dir) / "ffmpeg.exe"

if not target_alias.exists():
    shutil.copyfile(ffmpeg_exe, target_alias)

if ffmpeg_dir not in os.environ["PATH"]:
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

import whisper


def analyze_segment_acoustics(audio_path: str, start_sec: float, end_sec: float) -> str:
    """
    Reads a specific time window from the 16kHz mono WAV file and uses 
    acoustic heuristics (RMS Energy & Zero Crossing Rate) to classify 
    whether the speaker is the close-mic'd professor or a distant student.
    """
    audio_file = Path(audio_path)
    if not audio_file.exists():
        return "Professor"

    with wave.open(str(audio_file), 'rb') as wf:
        sample_rate = wf.getframerate()
        sampwidth = wf.getsampwidth()

        start_frame = int(start_sec * sample_rate)
        end_frame = int(end_sec * sample_rate)
        n_frames_to_read = end_frame - start_frame

        if n_frames_to_read <= 0:
            return "Professor"

        wf.setpos(max(0, start_frame))
        raw_bytes = wf.readframes(n_frames_to_read)

    if not raw_bytes:
        return "Professor"

    # Convert raw PCM bytes to NumPy float array (-1.0 to 1.0)
    if sampwidth == 2:
        audio_data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        audio_data = np.frombuffer(raw_bytes, dtype=np.int8).astype(np.float32) / 128.0

    if len(audio_data) == 0:
        return "Professor"

    # 1. Metric: RMS Energy (Proximity to microphone)
    rms_energy = np.sqrt(np.mean(audio_data**2))

    # 2. Metric: Zero Crossing Rate (Acoustic turbulence / room reverb scatter)
    zcr = np.mean(np.abs(np.diff(np.sign(audio_data))))

    # Lecture hall close-mic vs room boundary mic thresholds
    if rms_energy < 0.006 or zcr > 0.22:
        return "Other Person"
    
    return "Professor"


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

        print(f"[*] Transcribing: {media_path.name} (word timestamps & diarization enabled)...")
        
        result = self.model.transcribe(
            str(media_path),
            language=language,
            word_timestamps=True,
            verbose=False
        )

        segments = result.get("segments", [])
        if not segments:
            return []

        # Flatten words with timing
        all_words = []
        for segment in segments:
            for w in segment.get("words", []):
                all_words.append({
                    "word": w["word"].strip(),
                    "start": w["start"],
                    "end": w["end"]
                })

        if not all_words:
            return [{"start": segments[0]["start"], "end": segments[-1]["end"], "speaker": "Professor", "text": result.get("text", "")}]

        paragraphs = []
        current_words = []
        para_start = all_words[0]["start"]
        last_word_end = all_words[0]["end"]

        for item in all_words:
            word = item["word"]
            start = item["start"]
            end = item["end"]

            silence_gap = start - last_word_end

            if silence_gap >= pause_threshold_sec and current_words:
                # Classify speaker for the completed paragraph block
                speaker_tag = analyze_segment_acoustics(str(media_path), para_start, last_word_end)
                paragraphs.append({
                    "start": para_start,
                    "end": last_word_end,
                    "speaker": speaker_tag,
                    "text": " ".join(current_words)
                })
                current_words = [word]
                para_start = start
            else:
                current_words.append(word)

            last_word_end = end

        if current_words:
            speaker_tag = analyze_segment_acoustics(str(media_path), para_start, last_word_end)
            paragraphs.append({
                "start": para_start,
                "end": last_word_end,
                "speaker": speaker_tag,
                "text": " ".join(current_words)
            })

        return paragraphs

    def export_transcripts(self, paragraphs, output_dir, file_stem: str):
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        txt_file = out_dir / f"{file_stem}_transcript.txt"
        md_file = out_dir / f"{file_stem}_transcript.md"

        # Plain text export with speaker labels
        with open(txt_file, "w", encoding="utf-8") as f:
            for p in paragraphs:
                f.write(f"[{p['speaker']}] {p['text']}\n\n")

        # Formatted Markdown export with visual callouts for student Q&A
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(f"# Lecture Transcript: {file_stem}\n\n---\n\n")
            for p in paragraphs:
                ts = f"**[{self._format_timestamp(p['start'])} - {self._format_timestamp(p['end'])}]**"
                if "Student" in p["speaker"]:
                    f.write(f"{ts} 🎙️ *[Student Q&A]*:\n> {p['text']}\n\n")
                else:
                    f.write(f"{ts} **[Professor]**:\n{p['text']}\n\n")

        print(f"[+] Transcripts saved successfully:")
        print(f"    - Plaintext: {txt_file.resolve()}")
        print(f"    - Markdown:  {md_file.resolve()}")

        return txt_file, md_file