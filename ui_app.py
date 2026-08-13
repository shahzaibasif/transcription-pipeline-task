import os
import tempfile
import streamlit as st
import pandas as pd
import threading
import time

from pipeline.preprocess import preprocess_audio
from pipeline.transcriber import transcribe as pipeline_transcribe
from pipeline.exporter import save_output

try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except Exception:
    WhisperModel = None
    HAS_FASTER_WHISPER = False


st.set_page_config(page_title="Transcription Pipeline UI", layout="centered")


def _render_author_blocks(author_name: str = "Shahzaib", author_url: str | None = None, author_email: str | None = None, website: str | None = None):
    """Render a compact, professional author block inside the Streamlit sidebar.

    Places a concise attribution and short marketing line in the sidebar. Uses
    theme-aware CSS so it looks polished in both dark and light Streamlit themes.
    """
    # Build optional link fragments
    url_md = f" <a href=\"{author_url}\" target=\"_blank\">{author_url}</a>" if author_url else ""
    website_md = f" <a href=\"{website}\" target=\"_blank\">{website}</a>" if website else ""
    email_md = f" <a href=\"mailto:{author_email}\">{author_email}</a>" if author_email else ""

    css = """
    <style>
    .sidebar-author-card {
        display:flex;
        gap:10px;
        align-items:center;
        padding:12px;
        border-radius:10px;
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        box-shadow: 0 1px 4px rgba(16,24,40,0.04);
        color: #0f172a;
        margin-bottom: 10px;
    }
    .sidebar-avatar {
        width:56px;
        height:56px;
        border-radius:12px;
        background: #e6eef8;
        display:flex;
        align-items:center;
        justify-content:center;
        font-weight:700;
        color:#0f172a;
        flex-shrink:0;
        overflow:hidden;
    }
    .sidebar-avatar img{ width:100%; height:100%; object-fit:cover; }
    .sidebar-author-body{ font-size:13px; line-height:1.1; }
    .sidebar-author-name{ font-weight:700; color:inherit; margin-bottom:4px; }
    .sidebar-author-role{ font-size:12px; color:#64748b; margin-bottom:8px; }
    .sidebar-author-contacts{ display:flex; gap:8px; align-items:center; font-size:12px; }
    .contact-pill{ display:inline-flex; gap:6px; align-items:center; padding:6px 8px; border-radius:999px; background:rgba(15,23,42,0.03); text-decoration:none; color:inherit; }
    .contact-pill svg{ width:14px; height:14px; opacity:0.9 }

    @media (prefers-color-scheme: dark) {
        .sidebar-author-card{ background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); color:#e6eef8; box-shadow:none }
        .sidebar-avatar{ background:#0b1220; color:#e6eef8 }
        .sidebar-author-role{ color:#94a3b8 }
        .contact-pill{ background: rgba(255,255,255,0.03) }
    }
    </style>
    """

    st.sidebar.markdown(css, unsafe_allow_html=True)

    # Avatar: use provided URL if it's an image, otherwise show initials
    avatar_html = ""
    if author_url and author_url.endswith(('.png', '.jpg', '.jpeg', '.svg')):
        avatar_html = f"<div class=\"sidebar-avatar\"><img src=\"{author_url}\" alt=\"{author_name}\"/></div>"
    else:
        initials = "".join([p[0] for p in author_name.split()][:2]).upper()
        avatar_html = f"<div class=\"sidebar-avatar\">{initials}</div>"

    # Contacts: compose pills for website and email
    contacts = []
    if website:
        contacts.append(f"<a class=\"contact-pill\" href=\"{website}\" target=\"_blank\" aria-label=\"Website\">"
                        f"<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=1.5\">"
                        f"<path d=\"M14 3h7v7\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/><path d=\"M10 14L21 3\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/></svg> Website</a>")
    if author_email:
        contacts.append(f"<a class=\"contact-pill\" href=\"mailto:{author_email}\" aria-label=\"Email\">"
                        f"<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=1.5\">"
                        f"<path d=\"M3 8.5v7A2.5 2.5 0 0 0 5.5 18h13a2.5 2.5 0 0 0 2.5-2.5v-7\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>"
                        f"<path d=\"M21 7.5l-9 6-9-6\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/></svg> Email</a>")

    contacts_html = "".join(contacts)

    sidebar_html = (
        f"<div class=\"sidebar-author-card\">{avatar_html}"
        f"<div class=\"sidebar-author-body\"><div class=\"sidebar-author-name\">{author_name}</div>"
        f"<div class=\"sidebar-author-role\">Creator — Transcription Pipeline</div>"
        f"<div class=\"sidebar-author-contacts\">{contacts_html}</div></div></div>"
    )

    st.sidebar.markdown(sidebar_html, unsafe_allow_html=True)

    return None


def available_models():
    defaults = ["tiny", "base", "small", "medium",  "large"]
    # models_dir = os.path.join(os.getcwd(), "models")
    # if os.path.isdir(models_dir):
    #     for name in os.listdir(models_dir):
    #         if name not in defaults:
    #             defaults.append(name)
    return defaults


# Human-friendly model size hints (approximate)
MODEL_SPECS = {
    "tiny": "tiny (≈39 MB)",
    "base": "base (≈74 MB)",
    "small": "small (≈244 MB)",
    "medium": "medium (≈769 MB)",
    "large": "large (≈1550 MB)",
}

# Language code → name map for dropdown
LANG_MAP = {
    "auto": "Auto-detect",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "ur": "Urdu",
}


def load_model(model_name: str):
    key = f"whisper_model::{model_name}"
    if st.session_state.get("loaded_model_name") == model_name and st.session_state.get("loaded_model"):
        return st.session_state.get("loaded_model")

    model = None
    if HAS_FASTER_WHISPER:
        model = WhisperModel(model_name, device="cpu", compute_type="int8", download_root="models")
        st.session_state["loaded_model_name"] = model_name
        st.session_state["loaded_model"] = model

    return model


def transcribe_file(input_path: str, multilingual: bool, language: str, model_name: str | None = None):
    processed = preprocess_audio(input_path)

    # Prefer a locally loaded faster-whisper model when available
    if model_name and HAS_FASTER_WHISPER:
        model = load_model(model_name)
        if model is not None:
            segments, info = model.transcribe(processed, multilingual=multilingual, language=language)
            res = []
            txt = []
            for s in segments:
                txt.append(s.text)
                res.append({"start": s.start, "end": s.end, "text": s.text.strip()})
            result = {"language": info.language if hasattr(info, "language") else language,
                      "duration": res[-1]["end"] if res else 0,
                      "text": " ".join(txt).strip(),
                      "segments": res}
        else:
            result = pipeline_transcribe(processed, multilingual=multilingual, language=language)
    else:
        # Fallback to pipeline's transcribe implementation
        result = pipeline_transcribe(processed, multilingual=multilingual, language=language)
    try:
        if os.path.exists(processed):
            os.remove(processed)
    except Exception:
        pass
    return result


def main():
    st.title("Transcription Pipeline")


    # Author block displayed in the sidebar (professional attribution)
    _render_author_blocks(
        author_name="Shahzaib Asif",
        author_url="https://github.com/shahzaibasif",
        author_email="shahzaib.asif024@gmail.com",
        website="https://shahzaibasif.github.io",
    )

    st.write("Upload an audio file (WAV or MP3). The app will preprocess and transcribe it.")

    uploaded = st.file_uploader("Upload audio", type=["wav", "mp3"])

    # Place model, language, and multilingual checkbox on one row (model first)
    col_model, col_lang, col_check = st.columns([2, 2, 1])

    # Model selector (shows "model — size")
    with col_model:
        model_keys = available_models()
        model_labels = [f"{k} — {MODEL_SPECS.get(k, k)}" for k in model_keys]
        if model_labels:
            default_model_idx = model_keys.index("base") if "base" in model_keys else 0
            model_selection = st.selectbox("Model", options=model_labels, index=default_model_idx, key="ui_model")
            selected_model = model_selection.split(" ")[0]
        else:
            selected_model = st.selectbox("Model", options=["base", "small", "tiny"], index=0, key="ui_model_fallback")

    # Language selector (shows "code — name")
    with col_lang:
        lang_labels = [f"{code} — {name}" for code, name in LANG_MAP.items()]
        default_idx = list(LANG_MAP.keys()).index("en") if "en" in LANG_MAP else 0
        lang_selection = st.selectbox("Language", options=lang_labels, index=default_idx, key="ui_lang")
        language = lang_selection.split(" ")[0]

    # Multilingual checkbox on the same row (right-aligned)
    with col_check:
        multilingual = st.checkbox("Multilingual", value=False)

    if uploaded is not None:
        file_ext = os.path.splitext(uploaded.name)[1].lower()
        if file_ext not in (".wav", ".mp3"):
            st.error("Unsupported file type. Please upload WAV or MP3.")
            return

        tmp_dir = tempfile.gettempdir()
        tmp_path = os.path.join(tmp_dir, f"streamlit_upload{file_ext}")
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())

        st.audio(tmp_path)

        if st.button("Start Transcription"):
            # Run transcription in a background thread and show a progress bar
            result_container = {}

            def run_transcription():
                try:
                    res = transcribe_file(tmp_path, multilingual=multilingual, language=(None if language == "auto" else language), model_name=selected_model)
                    result_container["result"] = res
                except Exception as e:
                    result_container["error"] = str(e)

            thread = threading.Thread(target=run_transcription)
            thread.start()

            progress = st.progress(0)
            status = st.empty()
            status.text("Transcribing — this may take a while while the model runs...")

            # Animate progress while thread runs. Cap at 95% until complete.
            pct = 0
            while thread.is_alive():
                pct = min(pct + 5, 95)
                progress.progress(pct)
                time.sleep(0.2)

            thread.join()
            progress.progress(100)

            if "error" in result_container:
                st.error(f"Transcription failed: {result_container['error']}")
                return

            result = result_container.get("result")

            st.success("Transcription completed")

            st.subheader("Full Transcript")
            st.text_area("Transcript", value=result.get("text", ""), height=300)

            segments = result.get("segments", [])
            if segments:
                df = pd.DataFrame(segments)
                st.subheader("Segments")
                st.dataframe(df)

            st.download_button("Download transcript (txt)", data=result.get("text", ""), file_name="transcript.txt", mime="text/plain")

            if st.button("Save to output/ folder"):
                try:
                    save_output(result)
                    st.success("Saved to output/transcript.txt and output/transcript.json")
                except Exception as e:
                    st.error(f"Failed to save output: {e}")

        # cleanup temporary uploaded file
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()
