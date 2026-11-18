"""
Streamlit front end: give it a lecture video (file or link), get a study PDF back.
Run with:  streamlit run app.py
"""
import shutil

import streamlit as st

from main import run_pipeline
from pdf_builder import build_pdf
from video_source import ALLOWED_EXTENSIONS, download_video, save_upload
from workspace import new_run_dir

st.set_page_config(page_title="Lecture to PDF", page_icon="📝")
st.title("Lecture Video → Study PDF")
st.caption(
    "Upload a lecture video or paste a link. You'll get a PDF with the transcript, "
    "the board snapshots, and the writing on the board as text."
)

with st.sidebar:
    st.header("Settings")
    whisper_model = st.selectbox(
        "Speech recognition model",
        ["tiny", "base", "small"],
        index=1,
        help="Larger models transcribe more accurately but take longer.",
    )

mode = st.radio("Video source", ["Upload a file", "Paste a link"], horizontal=True)
uploaded, url = None, ""
if mode == "Upload a file":
    uploaded = st.file_uploader("Lecture video", type=[ext.lstrip(".") for ext in sorted(ALLOWED_EXTENSIONS)])
else:
    url = st.text_input("Video link", placeholder="https://www.youtube.com/watch?v=...")

ready = uploaded is not None if mode == "Upload a file" else bool(url.strip())
st.caption("Processing takes a few minutes for a short lecture, and longer for a full-length one.")

if st.button("Generate PDF", type="primary", disabled=not ready):
    st.session_state.pop("result", None)
    run_dir = new_run_dir()  # temporary; holds the video, frames and every intermediate file
    try:
        with st.status("Working on your lecture...", expanded=True) as status:
            if mode == "Upload a file":
                st.write("Saving your upload...")
                video = save_upload(uploaded, uploaded.name, dest_dir=run_dir / "raw")
            else:
                st.write("Downloading the video...")
                video = download_video(url.strip(), dest_dir=run_dir / "raw")

            storyboard = run_pipeline(
                str(video.path),
                whisper_model=whisper_model,
                title=video.title,
                source=video.source,
                progress=lambda message: st.write(message.strip()),
                work_dir=run_dir,
            )

            st.write("Building the PDF...")
            pdf_bytes = build_pdf(storyboard).read_bytes()
            status.update(label="Done", state="complete", expanded=False)

        # Only the PDF survives, in memory, for the download button
        st.session_state["result"] = {"title": video.title, "name": f"{video.path.stem}.pdf", "bytes": pdf_bytes}
    except (ValueError, RuntimeError) as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Something went wrong while processing the video: {e}")
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

result = st.session_state.get("result")
if result:
    st.success(f"Your PDF for “{result['title']}” is ready.")
    st.download_button(
        "Download PDF", data=result["bytes"], file_name=result["name"], mime="application/pdf", type="primary"
    )
