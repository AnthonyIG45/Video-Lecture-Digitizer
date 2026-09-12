# SynchroBoard: Multimodal Lecture Video Digitization & Audio-Visual Alignment

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/Architecture-Modular%20Pipeline-brightgreen.svg)]()

> **Senior Capstone Proof-of-Concept**  
> An automated pipeline that transforms raw chalk/whiteboard lecture recordings into an interactive, indexed mathematical storyboard. By decoupling visual stroke stabilization from speech transcription, SynchroBoard isolates newly written formulas, transcribes spoken explanations, and synchronizes math notation with timestamped lecture playback.

---

## 📑 Table of Contents
- [Problem Overview](#problem-overview)
- [System Architecture](#system-architecture)
- [Core Pipeline Modules](#core-pipeline-modules)
  - [1. Visual Extraction & Stroke Stability](#1-visual-extraction--stroke-stability)
  - [2. Audio Processing & Phoneme Alignment](#2-audio-processing--phoneme-alignment)
  - [3. Graphic Recognition & LaTeX Transduction](#3-graphic-recognition--latex-transduction)
  - [4. Multimodal Temporal Synchronization Engine](#4-multimodal-temporal-synchronization-engine)
  - [5. Interactive Frontend (HCI)](#5-interactive-frontend-hci)
- [Project Directory Structure](#project-directory-structure)
- [Installation & Setup](#installation--setup)
- [Usage & Pipeline Execution](#usage--pipeline-execution)
- [Configuration Reference](#configuration-reference)
- [Verification & Quality Control](#verification--quality-control)
- [Roadmap & Milestone Tracking](#roadmap--milestone-tracking)
- [License & Acknowledgments](#license--acknowledgments)

---

## 🎯 Problem Overview

STEM lecture videos present a dual-modality cognitive load: students struggle to simultaneously copy dense whiteboard derivations and follow spoken conceptual explanations. Traditional optical character recognition (OCR) fails on raw lecture video due to:
1. **Dynamic Lecturer Occlusion:** The instructor continually writes, steps in front of derivations, and introduces rapid motion artifacts.
2. **Stroke Instability:** Writing is an incremental process. Extracting video frames mid-stroke produces fragmented, unintelligible equations.
3. **Temporal Lead/Lag Discrepancies:** Instructors rarely speak math symbols simultaneously with writing. Explanations either precede writing (prospective) or follow completion (retrospective).

**SynchroBoard** solves these challenges by combining background subtraction and person masking with stroke stability buffers, neural LaTeX transduction, and asymmetric temporal alignment.

---

## 🏗 System Architecture

```
                    +------------------------------------+
                    |     Raw Lecture Video (.mp4)       |
                    +------------------------------------+
                                      |
                  +-------------------+-------------------+
                  |                                       |
                  v                                       v
        [ 1. Video Pipeline ]                   [ 2. Audio Pipeline ]
       • Background subtraction                • Audio track extraction (ffmpeg)
       • Lecturer occlusion mask (MediaPipe)   • Whisper / WhisperX transcription
       • Edge/Stroke stability buffer          • Word/Phoneme-level timestamps
                  |                                       |
                  v                                       |
      [ 3. Graphic Recognition ]                          |
       • Patch classification (3-Way)                     |
         (Formula vs. Diagram vs. Text)                   |
       • LaTeX-OCR transduction (pix2tex)                 |
       • Syntax confidence gate (SymPy)                   |
                  |                                       |
                  +-------------------+-------------------+
                                      |
                                      v
                      [ 4. Synchronization Engine ]
                       • Asymmetric temporal window clustering
                       • Perceptual hashing (pHash) deduplication
                                      |
                                      v
                       [ 5. Interactive UI (HCI) ]
                        • Bidirectional synchronized playback
                        • Searchable KaTeX math index
                        • Clickable derivation storyboard
```

---

## 🔬 Core Pipeline Modules

### 1. Visual Extraction & Stroke Stability
* **Lecturer Occlusion Masking:** Uses MediaPipe Pose / Selfie Segmentation to generate a real-time binary mask over the instructor, restricting analysis purely to the exposed board area.
* **MOG2 Adaptive Background Modeling:** Tracks persistent stroke additions while filtering out high-frequency illumination flicker and minor sensor noise.
* **Temporal Stability Accumulator:** A region of interest (ROI) is flagged as an active stroke event only when its Canny/Sobel gradient variance remains below an $\epsilon$-threshold for $>2.5$ seconds following occlusion clearance.

### 2. Audio Processing & Phoneme Alignment
* **Track Extraction:** Native Demuxing using FFmpeg to high-fidelity 16kHz mono WAV.
* **Transcription & Alignment:** Utilizes `whisperx` (Whisper backbone + wav2vec 2.0 forced alignment) to generate word-level timestamp boundaries ($[t_{\text{start}}, t_{\text{end}}]$).
* **Segment Boundary Parsing:** Assembles words into coherent utterance units while preserving sub-second token boundaries.

### 3. Graphic Recognition & LaTeX Transduction
* **3-Way Region Classifier:** Distinguishes between standalone mathematical expressions, freeform diagrams, and running blackboard text using stroke density and aspect ratio features (MobileNetV3 backbone).
* **LaTeX-OCR Transduction:** Formula crops pass to `pix2tex` for sequence-to-sequence visual-to-LaTeX token generation.
* **Syntactic Confidence Gate:** Transduced LaTeX strings are evaluated inside a `sympy.parsing.latex` boundary. Ill-formed derivations are caught and either fall back to raw crops or trigger confidence penalties without breaking pipeline state.

### 4. Multimodal Temporal Synchronization Engine
* **Perceptual Deduplication:** Stable crops are hashed via `imagehash.phash`. Hamming distances ($\Delta_{\text{hash}} < 8\text{ bits}$) are clustered to prevent repeated registrations of static formulas.
* **Asymmetric Temporal Windowing:** Maps visual completion timestamps ($t_{\text{visual}}$) to speech chunks across an asymmetric boundary:
  $$\mathcal{W}(t_{\text{visual}}) = [t_{\text{visual}} - \Delta t_{\text{lead}},\; t_{\text{visual}} + \Delta t_{\text{lag}}]$$
  *(Default: $\Delta t_{\text{lead}} = 3.0\text{s}$, $\Delta t_{\text{lag}} = 10.0\text{s}$).*

### 5. Interactive UI (HCI)
* Built using a responsive frontend architecture (Streamlit / Vite-React + HTML5 video control API).
* Interactive storyboard cards containing the raw crop, rendered KaTeX equation, and synchronized transcript.
* Bidirectional seeking: Clicking any formula instantly seeks the video to the teacher's explanation.

---

## 📂 Project Directory Structure

```plaintext
synchroboard/
├── data/
│   ├── raw/                  # Input lecture videos (.mp4)
│   ├── processed/            # Cropped board regions & audio tracks
│   └── output/               # Output JSON storyboards and LaTeX dumps
├── configs/
│   └── pipeline_config.yaml  # Thresholds, model weights, and temporal windows
├── src/
│   ├── video/
│   │   ├── segmenter.py      # MediaPipe occlusion masking
│   │   ├── stability.py      # Temporal stroke buffer & MOG2 subtraction
│   │   └── cropper.py        # Contour discovery & bounding box normalization
│   ├── audio/
│   │   ├── extractor.py      # FFmpeg demuxing wrappers
│   │   └── transcriber.py    # WhisperX pipeline & forced alignment
│   ├── recognition/
│   │   ├── classifier.py     # 3-way patch classification (Formula/Diagram/Text)
│   │   ├── latex_ocr.py      # pix2tex inference wrapper
│   │   └── verifier.py       # SymPy syntax confidence gate
│   ├── sync/
│   │   ├── deduplicator.py   # pHash Hamming clustering
│   │   └── aligner.py        # Temporal window audio-visual matcher
│   └── ui/
│       ├── app.py            # Streamlit / Web UI entrypoint
│       └── components/       # Custom video player & KaTeX renderers
├── tests/
│   ├── test_stability.py
│   ├── test_latex_parser.py
│   └── test_sync_window.py
├── requirements.txt
├── Makefile
└── README.md
```

---

## ⚡ Installation & Setup

### Prerequisites
* Python 3.10 or higher
* FFmpeg (must be available on system `$PATH`)
* CUDA-capable GPU (recommended for WhisperX and pix2tex inference)

### Setup Instructions

```bash
# 1. Clone repository
git clone https://github.com/your-username/synchroboard.git
cd synchroboard

# 2. Initialize virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# 3. Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install system binaries (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install -y ffmpeg

# 4. (Alternative: Windows via Scoop / Chocolatey)
# choco install ffmpeg
```

---

## 🚀 Usage & Pipeline Execution

### Running the End-to-End Pipeline
Execute the full extraction and alignment pipeline on a raw recording:

```bash
python -m src.sync.aligner \
    --video-path data/raw/calculus_lecture_sample.mp4 \
    --config configs/pipeline_config.yaml \
    --output-dir data/output/run_01
```

### Launching the Interactive Interface
Once alignment metadata is generated, run the visualization storyboard:

```bash
streamlit run src/ui/app.py -- --manifest data/output/run_01/storyboard.json
```

---

## ⚙️ Configuration Reference (`pipeline_config.yaml`)

```yaml
video:
  sample_fps: 5
  stability_duration_sec: 2.5
  canny_threshold_low: 50
  canny_threshold_high: 150
  min_contour_area: 1200

audio:
  model_size: "medium"
  compute_type: "float16"
  batch_size: 8

recognition:
  confidence_threshold: 0.82
  enforce_sympy_verification: true

sync:
  lead_window_sec: 3.0
  lag_window_sec: 10.0
  phash_hamming_threshold: 8
```

---

## 🧪 Verification & Quality Control

Unit and integration tests validate the stroke stability detector and LaTeX syntax verifier:

```bash
# Run test suite
pytest tests/ -v

# Run syntax gate sanity checks
python -m tests.test_latex_parser
```

---

## 🗺 Roadmap & Milestone Tracking

- [x] **Milestone 1: Pipeline Specification & Feasibility Audit**
  - Architecture blueprinting and multimodal pipeline mapping.
- [ ] **Milestone 2: Video Extraction & Stroke Stability PoC**
  - MediaPipe person segmentation + MOG2 temporal variance stabilization.
- [ ] **Milestone 3: Transduction & Audio-Visual Alignment**
  - WhisperX phoneme timestamp integration and `pix2tex` LaTeX OCR extraction.
- [ ] **Milestone 4: Synchronized Storyboard UI**
  - Bidirectional video-to-formula player in Streamlit with KaTeX rendering.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
