"""
Builds a study PDF from a storyboard.json (layout documented in storyboard.py).

Deliberately standalone: it reads only the JSON and the image files it points to, and imports nothing
from the video/audio pipeline. Each section shows the transcript paragraph, the board snapshot image,
and the OCR'd board text with formulas typeset where they can be rendered.
"""
import io
import json
import re
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer

MATH_DPI = 200
MATH_FONT_SIZE = 13
MAX_SNAPSHOT_HEIGHT = 3.4 * inch

_DISPLAY_MATH = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)


def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("Title", parent=base["Title"], fontSize=22, spaceAfter=4),
        "subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontSize=9, textColor=colors.grey, spaceAfter=10),
        "time": ParagraphStyle("Time", parent=base["Normal"], fontSize=8.5, textColor=colors.HexColor("#555555"),
                               fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=2),
        "body": ParagraphStyle("Body", parent=base["Normal"], fontSize=10.5, leading=14.5, spaceAfter=6),
        "label": ParagraphStyle("Label", parent=base["Normal"], fontSize=8.5, textColor=colors.HexColor("#555555"),
                                spaceBefore=4, spaceAfter=3),
        "mono": ParagraphStyle("Mono", parent=base["Code"], fontSize=8, leading=10, textColor=colors.HexColor("#444444"),
                               backColor=colors.HexColor("#F3F3F3"), borderPadding=3, spaceAfter=4),
    }


def _normalize_latex(latex: str) -> list:
    """Rewrites OCR LaTeX into what matplotlib's math renderer accepts; returns one formula per line."""
    latex = re.sub(r"\\(?:Bigg|bigg|Big|big)[lr]?", "", latex)           # sizing prefixes are unsupported
    latex = re.sub(r"\\hspace\{[^}]*\}", " ", latex)
    latex = re.sub(r"\\q?quad", " ", latex)
    latex = re.sub(r"\\(?:underbrace|boxed|fbox|underbar)", "", latex)    # keep the content, drop the wrapper
    latex = re.sub(r"\\(?:display|text|script|scriptscript)style", "", latex)  # style switches are unsupported
    latex = re.sub(r"\\(?:begin|end)\{[a-zA-Z*]+\}", "", latex)           # matrix/aligned environments

    lines = [part.replace("&", " ").strip() for part in latex.split("\\\\")]
    return [line for line in lines if line]


def _render_math(latex: str, max_width: float):
    """Renders one formula to a reportlab Image, or returns None if it can't be parsed."""
    try:
        fig = Figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${latex}$", fontsize=MATH_FONT_SIZE)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=MATH_DPI, bbox_inches="tight", pad_inches=0.04)
    except Exception:
        return None

    buf.seek(0)
    px_w, px_h = PILImage.open(buf).size
    width, height = px_w * 72 / MATH_DPI, px_h * 72 / MATH_DPI
    if width > max_width:
        height *= max_width / width
        width = max_width
    buf.seek(0)
    return Image(buf, width=width, height=height, hAlign="LEFT")


def _is_garbled_prose(prose: str) -> bool:
    """OCR that mistook handwriting for math leaks LaTeX syntax into plain text; real prose has none."""
    return bool(re.search(r"\\[a-zA-Z]+|[{}^_]", prose))


def _ocr_flowables(text: str, styles: dict, max_width: float) -> list:
    """
    Converts OCR output (prose with $$...$$ display formulas) into flowables. Anything that is
    clearly broken (LaTeX leaking into prose, formulas that cannot be typeset) is left out; the
    board snapshot above it always shows the real content.
    """
    flowables = []
    for i, segment in enumerate(_DISPLAY_MATH.split(text)):
        if i % 2 == 0:  # prose; inline $..$ is shown as plain text
            prose = " ".join(segment.replace("$", "").split())
            if prose and not _is_garbled_prose(prose):
                flowables.append(Paragraph(escape(prose), styles["body"]))
            continue

        for formula in _normalize_latex(segment):
            rendered = _render_math(formula, max_width)
            if rendered is not None:
                flowables += [rendered, Spacer(1, 4)]
    return flowables


def _snapshot_flowables(snapshot: dict, base_dir: Path, styles: dict, frame_w: float) -> list:
    head = [Paragraph(f"Board snapshot at {_fmt_time(snapshot['time'])}", styles["label"])]

    image_path = base_dir / snapshot["image"]
    if image_path.exists():
        with PILImage.open(image_path) as im:
            px_w, px_h = im.size
        scale = min(frame_w / px_w, MAX_SNAPSHOT_HEIGHT / px_h)
        head.append(Image(str(image_path), width=px_w * scale, height=px_h * scale, hAlign="LEFT"))
    else:
        head.append(Paragraph("(snapshot image not found)", styles["mono"]))

    flowables = [KeepTogether(head)]  # never leave the caption stranded away from its image

    ocr_flowables = []
    for block in snapshot.get("blocks", []):
        if block.get("text"):
            ocr_flowables += _ocr_flowables(block["text"], styles, frame_w)
    if ocr_flowables:
        flowables.append(Paragraph("Board text (OCR)", styles["label"]))
        flowables += ocr_flowables
    return flowables


def build_pdf(storyboard_path, output_path=None) -> Path:
    """Reads storyboard.json and writes the PDF. Returns the PDF path (default: next to the JSON)."""
    storyboard_path = Path(storyboard_path)
    data = json.loads(storyboard_path.read_text(encoding="utf-8"))
    base_dir = storyboard_path.parent
    output_path = Path(output_path) if output_path else storyboard_path.with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    title = data.get("title") or storyboard_path.stem
    doc = SimpleDocTemplate(
        str(output_path), pagesize=letter, title=title,
        leftMargin=0.8 * inch, rightMargin=0.8 * inch, topMargin=0.8 * inch, bottomMargin=0.8 * inch,
    )
    frame_w = doc.width
    styles = _styles()

    story = [Paragraph(escape(title), styles["title"])]
    if data.get("source"):
        story.append(Paragraph(f"Source: {escape(data['source'])}", styles["subtitle"]))
    story.append(HRFlowable(width="100%", color=colors.lightgrey, spaceAfter=4))

    sections = data.get("sections", [])
    if not sections:
        story.append(Paragraph("No transcript or board content was extracted from this video.", styles["body"]))

    # Only tag sections whose speaker differs from the dominant one; a label that applies to every
    # section (e.g. a lone lecturer the speaker heuristic mislabels) is noise
    speaker_counts = Counter(s.get("speaker") for s in sections)
    dominant_speaker = speaker_counts.most_common(1)[0][0] if speaker_counts else None

    for section in sections:
        header = [Paragraph(f"{_fmt_time(section['start'])} – {_fmt_time(section['end'])}", styles["time"])]
        speaker = section.get("speaker")
        text = escape(section.get("text", ""))
        if speaker and speaker != dominant_speaker:
            text = f"<i>[{escape(speaker)}]</i> {text}"
        if text:
            header.append(Paragraph(text, styles["body"]))
        story.append(KeepTogether(header))

        for snapshot in section.get("snapshots", []):
            story += _snapshot_flowables(snapshot, base_dir, styles, frame_w)
            story.append(Spacer(1, 8))

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(letter[0] / 2, 0.45 * inch, f"{title}  ·  page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output_path
