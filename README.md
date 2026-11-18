# SynchroBoard: Multimodal Lecture Video Digitization & Audio-Visual Alignment

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/Architecture-Modular%20Pipeline-brightgreen.svg)]()

> **Senior Capstone Proof-of-Concept**  
> An automated pipeline that turns a recorded chalk/whiteboard lecture into a study PDF. Give it a video file or a YouTube link; it transcribes the speech, captures each new state of the board, reads the writing on it, and lays all of it out in time order.

---

## 📑 Table of Contents
- [Problem Overview](#-problem-overview)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Privacy & Storage](#-privacy--storage)
- [Known Limitations](#-known-limitations)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 🎯 Problem Overview

STEM lecture videos present a dual-modality cognitive load: students struggle to simultaneously copy dense whiteboard derivations and follow spoken conceptual explanations. Traditional optical character recognition (OCR) fails on raw lecture video due to:
1. **Dynamic Lecturer Occlusion:** The instructor continually writes, steps in front of derivations, and introduces rapid motion artifacts.
2. **Stroke Instability:** Writing is an incremental process. Extracting video frames mid-stroke produces fragmented, unintelligible equations.
3. **Temporal Lead/Lag Discrepancies:** Instructors rarely speak math symbols simultaneously with writing. Explanations either precede writing (prospective) or follow completion (retrospective).

**SynchroBoard** addresses these by capturing the board only when its writing has reached a new state (and, when the lecturer's hand never leaves the frame, by picking the clearest recent frame), segmenting each snapshot into lines, recognizing text and LaTeX per line, and placing every snapshot under the speech it accompanies.

---

## ⚙️ How It Works

```
        Video file or link (yt-dlp)
                    |
      +-------------+--------------+
      v                            v
 16 kHz mono audio            4 fps normalized video
      |                            |
      v                            v
 Whisper transcription       Keyframe extraction (ink-based)
 (word timestamps,           - board changes judged on the ink
  pause-based paragraphs)      mask, not raw pixels
      |                      - "settled" trigger + "progress"
      |                        trigger for hand-in-frame video
      |                            |
      |                            v
      |                      Block segmentation (OpenCV XY-cut)
      |                            |
      |                            v
      |                      Pix2Text OCR: prose + LaTeX
      |                            |
      +-------------+--------------+
                    v
             storyboard.json          <-- the only hand-off
                    |
         +----------+-----------+
         v                      v
   pdf_builder.py        app.py (Streamlit)
```

1. **Audio:** FFmpeg (bundled through `imageio-ffmpeg`) extracts 16 kHz mono audio; `openai-whisper` produces word-level timestamps that are grouped into paragraphs at pauses.
2. **Keyframes** (`extractor.py`, `ink.py`): the board region is analyzed as an ink mask that is robust to lighting flicker and noise. A snapshot is saved when the ink has stayed unchanged for 3 s, or, if the writing never settles, when 20 s have passed and the board has changed by at least 10%. In the latter case the least-occluded recent frame (smallest hand/pen blob) is used. Blank boards and unchanged states are skipped.
3. **Segmentation** (`board_segmenter.py`): classical OpenCV only. Ink is binarized, UI chrome and clipped edge fragments are dropped, and a recursive XY-cut splits the board at whitespace gaps into line-level blocks; stacked fractions are re-joined.
4. **OCR** (`block_ocr.py`): Pix2Text reads each block as mixed prose and LaTeX, then number-spacing artifacts are cleaned up.
5. **Storyboard** (`storyboard.py`): every keyframe is placed under the latest transcript paragraph that started at or before it, and everything is written to `storyboard.json`.
6. **PDF** (`pdf_builder.py`): ReportLab layout with the transcript, the board snapshot, and the OCR text with formulas typeset. Text that is clearly broken is left out; the snapshot always shows the real board.

The PDF builder and the UI know nothing about the processing pipeline. They only read `storyboard.json` (layout documented in [storyboard.py](storyboard.py)), so either can be replaced independently.

---

## 📂 Project Structure

```plaintext
Video-Lecture-Digitizer/
├── app.py               # Streamlit UI: upload a file or paste a link, get a PDF
├── main.py              # run_pipeline(): video -> storyboard.json (also runnable directly)
├── video_source.py      # Saves an upload or downloads a link (yt-dlp) to a local file
├── video_processor.py   # FFmpeg: audio extraction and 4 fps normalization
├── extractor.py         # Board ROI detection and ink-based keyframe extraction
├── ink.py               # Shared ink-mask / background-flattening helpers
├── board_segmenter.py   # Keyframe -> line-level blocks (XY-cut)
├── block_ocr.py         # Pix2Text text + LaTeX recognition per block
├── speech_to_text.py    # Whisper transcription and Markdown storyboard export
├── storyboard.py        # storyboard.json contract, keyframe placement
├── pdf_builder.py       # storyboard.json -> PDF
├── workspace.py         # Per-instance temporary storage that is deleted on exit
├── run_segmenter.py     # Dev helper: segment a processed keyframe folder
├── run_ocr.py           # Dev helper: OCR a segmented keyframe folder
├── .streamlit/config.toml
├── requirements.txt
└── data/
    ├── raw/             # Source videos used for command-line runs
    └── processed/       # Command-line outputs (git-ignored)
```

---

## ⚡ Installation & Setup

### Prerequisites
* Python 3.10 or higher (developed on 3.11)
* No separate FFmpeg install is needed; a copy is bundled through `imageio-ffmpeg`
* CPU is enough. A GPU is not required

### Setup

```bash
# 1. Clone
git clone https://github.com/AnthonyIG45/Video-Lecture-Digitizer.git
cd Video-Lecture-Digitizer

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # On Windows: .\.venv\Scripts\activate

# 3. Install dependencies (CPU build of PyTorch is pulled from the PyTorch index)
pip install -r requirements.txt
```

The first run downloads model weights (Whisper from OpenAI, Pix2Text from Hugging Face), roughly a few hundred MB in total. These are cached for later runs.

`rapidocr` and `onnxruntime` are pinned in `requirements.txt` on purpose: newer versions break Pix2Text's imports alongside PyTorch 2.4.1.

---

## 🚀 Usage

### Web app

```bash
streamlit run app.py
```

Choose **Upload a file** or **Paste a link** (YouTube and other sites supported by yt-dlp; single videos up to 60 minutes), pick a Whisper model size, and press **Generate PDF**. A short lecture takes a few minutes on CPU (about 2 to 3 minutes for a 4-minute video in testing; OCR and Whisper dominate).

### Command line

Set `SOURCE_VIDEO` at the bottom of `main.py` and run:

```bash
python main.py
```

This writes intermediate files under `data/processed/` and a `storyboard.json` under `data/processed/storyboards/`. Build a PDF from it with:

```python
from pdf_builder import build_pdf
build_pdf("data/processed/storyboards/<video-name>.json")
```

---

## 🔒 Privacy & Storage

The web app keeps nothing from your video on disk. Each run works in a temporary folder that is deleted as soon as the PDF is built, and every instance's working directory is removed when the app exits (leftovers from a hard-killed instance are swept on the next start). The finished PDF lives in memory for the download button only. Only model weights are cached between runs.

---

## ⚠️ Known Limitations

* **Handwritten prose** is often misread, and clipped lines at the edge of the frame are garbled. Handwritten formulas are much better, and the board image is always included in the PDF.
* **Figures are not detected.** Diagrams and graphs appear in the board snapshot, but are not cropped out or described, and the OCR silently skips them.
* **Capture heuristics** are tuned on a few videos. When a hand covers the board for long stretches, a snapshot may still include it, and a snapshot can catch a page mid-flip.
* **Speaker labels** come from a simple loudness heuristic and are unreliable; the PDF only shows a label when the speaker differs from the dominant one.
* **Board detection** falls back to the full frame when no clear rectangular board is found (typical for screen recordings and close-up paper shots).

---

## 🗺 Roadmap

- [x] **Milestone 1: Pipeline Specification & Feasibility Audit**
- [x] **Milestone 2: Video Extraction & Keyframe Capture PoC**
  - Ink-based stability detection and hand-aware progress snapshots (replaced the originally planned MediaPipe/MOG2 approach).
- [x] **Milestone 3: Transcription, OCR & Alignment**
  - Whisper word timestamps, line segmentation, Pix2Text LaTeX OCR, keyframe-to-speech placement.
- [x] **Milestone 4: Study PDF & Web UI**
  - Streamlit front end with upload/link input; PDF generated from a decoupled `storyboard.json`.
- [ ] **Future work:** figure detection and cropping, SymPy validation of OCR'd LaTeX, perceptual-hash deduplication, lecturer masking, bidirectional video seeking.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
