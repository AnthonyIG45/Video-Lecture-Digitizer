"""
Turns "a video the user gave us" into a local file the pipeline can read: either a saved upload or a
download from a link (YouTube and anything else yt-dlp supports). Knows nothing about the pipeline.
"""
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import imageio_ffmpeg
import yt_dlp

RAW_DIR = Path("data/raw")
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


@dataclass
class VideoInfo:
    path: Path
    title: str
    source: str  # the original URL or uploaded filename


def is_url(text: str) -> bool:
    parsed = urlparse(text.strip())
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")[:60] or "video"


def save_upload(fileobj, filename: str, dest_dir=RAW_DIR) -> VideoInfo:
    """Saves an uploaded file-like object under a sanitized name and returns its VideoInfo."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Use one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{_slug(Path(filename).stem)}{ext}"
    with open(dest, "wb") as out:
        shutil.copyfileobj(fileobj, out)
    return VideoInfo(path=dest, title=Path(filename).stem.replace("_", " "), source=filename)


def download_video(url: str, dest_dir=RAW_DIR, max_height: int = 720, max_duration_sec: int = 3600) -> VideoInfo:
    """Downloads a video (single video only, capped at max_height) and returns its VideoInfo."""
    if not is_url(url):
        raise ValueError("That doesn't look like a valid http(s) link.")

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    opts = {
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "cachedir": False,  # don't keep yt-dlp's own cache on disk either
        "format": f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b",
        "merge_output_format": "mp4",
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        "outtmpl": str(dest_dir / "yt_%(id)s.%(ext)s"),
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            meta = ydl.extract_info(url, download=False)
            duration = meta.get("duration") or 0
            if duration > max_duration_sec:
                raise ValueError(
                    f"This video is {duration // 60} minutes long; the limit is {max_duration_sec // 60} minutes. "
                    "Try a shorter video or upload a clip."
                )
            info = ydl.extract_info(url, download=True)
            path = Path(info["requested_downloads"][0]["filepath"])
    except yt_dlp.utils.DownloadError as e:
        raise RuntimeError(f"Could not download that video: {e}") from e

    return VideoInfo(path=path, title=info.get("title") or path.stem, source=url)
