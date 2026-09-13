import os
import shutil
import subprocess
from pathlib import Path
import imageio_ffmpeg

# Resolve bundled ffmpeg binary
FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

# Ensure the directory is on PATH for child processes
ffmpeg_dir = str(Path(FFMPEG_BIN).parent)
if ffmpeg_dir not in os.environ["PATH"]:
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

# Ensure alias 'ffmpeg.exe' exists in that folder
target_alias = Path(ffmpeg_dir) / "ffmpeg.exe"
if not target_alias.exists():
    shutil.copyfile(FFMPEG_BIN, target_alias)


class VideoProcessor:
    """Handles whole-video downsampling and 16kHz audio extraction."""

    @staticmethod
    def normalize_video(
        input_path: str,
        output_path: str,
        target_fps: int = 4,
        max_width: int = 1280
    ) -> Path:
        """
        Condenses video to low FPS (for vision/OCR) and standard resolution.
        """
        src = Path(input_path)
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            raise FileNotFoundError(f"Input video does not exist: {src.resolve()}")

        vf_filter = f"fps={target_fps},scale='min({max_width},iw)':-2"

        cmd = [
            FFMPEG_BIN, "-y",
            "-i", str(src),
            "-vf", vf_filter,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            "-an",  # Strip audio here to save space; audio is handled separately
            str(dest)
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return dest

    @staticmethod
    def extract_audio_for_stt(input_path: str, output_path: str) -> Path:
        """
        Extracts 16kHz 16-bit mono PCM WAV across the entire video for Whisper.
        """
        src = Path(input_path)
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            raise FileNotFoundError(f"Input video does not exist: {src.resolve()}")

        cmd = [
            FFMPEG_BIN, "-y",
            "-i", str(src),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(dest)
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return dest