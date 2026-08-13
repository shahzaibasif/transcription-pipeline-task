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


def _render_author_blocks(author_name: str = "Shahzaib", author_url: str | None = None, author_email: str | None = None, website: str | None = None, repo_url: str | None = None):
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
    /* Theme-aware profile card colors */
    :root,
    [data-theme="light"] {
        --card-bg: #ffffff;
        --card-text: #111827;
        --muted: #475569;
        --avatar-bg: linear-gradient(135deg, #dbeafe 0%, #e2e8f0 100%);
        --repo-bg: #f8fafc;
        --repo-icon-bg: #111827;
        --pill-bg: #f3f4f6;
        --card-border: rgba(148, 163, 184, 0.35);
        --card-shadow: rgba(15, 23, 42, 0.06);
    }

    [data-theme="dark"], .dark {
        --card-bg: #1f2937;
        --card-text: #f8fafc;
        --muted: #cbd5e1;
        --avatar-bg: linear-gradient(135deg, rgba(59,130,246,0.25), rgba(30,41,59,0.8));
        --repo-bg: #111827;
        --repo-icon-bg: #0f172a;
        --pill-bg: rgba(148,163,184,0.12);
        --card-border: rgba(148,163,184,0.22);
        --card-shadow: rgba(2,6,23,0.24);
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --card-bg: #1f2937;
            --card-text: #f8fafc;
            --muted: #cbd5e1;
            --avatar-bg: linear-gradient(135deg, rgba(59,130,246,0.25), rgba(30,41,59,0.8));
            --repo-bg: #111827;
            --repo-icon-bg: #0f172a;
            --pill-bg: rgba(148,163,184,0.12);
            --card-border: rgba(148,163,184,0.22);
            --card-shadow: rgba(2,6,23,0.24);
        }
    }

    .sidebar-author-card {
        display:flex;
        gap:10px;
        align-items:center;
        padding:12px;
        border-radius:12px;
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        box-shadow: 0 8px 20px var(--card-shadow);
        color: var(--card-text);
        margin-bottom: 10px;
    }
    .sidebar-avatar {
        width:56px;
        height:56px;
        border-radius:12px;
        background: var(--avatar-bg);
        display:flex;
        align-items:center;
        justify-content:center;
        font-weight:700;
        color:var(--card-text);
        flex-shrink:0;
        overflow:hidden;
    }
    .sidebar-avatar img{ width:100%; height:100%; object-fit:cover; }
    .sidebar-author-body{ font-size:13px; line-height:1.1; }
    .sidebar-author-name{ font-weight:700; color:inherit; margin-bottom:4px; }
    .sidebar-author-role{ font-size:12px; color:var(--muted); margin-bottom:8px; }
    .sidebar-author-contacts{ display:flex; gap:8px; align-items:center; font-size:12px; flex-wrap:wrap; }
    .contact-pill{ display:inline-flex; gap:6px; align-items:center; padding:6px 8px; border-radius:999px; background:var(--pill-bg); border: 1px solid var(--card-border); text-decoration:none; color:var(--card-text); }
    .contact-pill svg{ width:14px; height:14px; opacity:0.9 }

    /* Repository card styles */
    .sidebar-repo-card{ padding:12px; border-radius:12px; background:var(--repo-bg); border: 1px solid var(--card-border); box-shadow: 0 8px 20px var(--card-shadow); margin-top:6px; color:var(--card-text); }
    .repo-row{ display:flex; gap:10px; align-items:flex-start }
    .repo-icon{ width:40px; height:40px; border-radius:8px; background:var(--repo-icon-bg); display:flex; align-items:center; justify-content:center; color:white; flex-shrink:0 }
    .repo-body{ font-size:13px }
    .repo-title{ font-weight:600; margin-bottom:4px; font-size:13px }
    .repo-desc{ color:var(--muted); font-size:12px; font-weight:300 }

    .sidebar-repo-card a{ color:inherit; text-decoration:none }
    .sidebar-repo-card a:hover{ text-decoration:underline }

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

    # Contacts: compose pills for website, repo and email
    contacts = []
    if website:
        contacts.append(f"<a class=\"contact-pill\" href=\"{website}\" target=\"_blank\" aria-label=\"Website\">"
                        f"<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=1.5\">"
                        f"<path d=\"M14 3h7v7\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/><path d=\"M10 14L21 3\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/></svg> Website</a>")
    # Repo will be shown in its own card below; do not include it in contact pills
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

    # Render repository card below the author block if provided
    if repo_url:
        # derive a short repo display name from the URL
        try:
            repo_display = repo_url.split("github.com/")[-1].rstrip("/")
        except Exception:
            repo_display = repo_url

        repo_card = (
            f"<div class=\"sidebar-repo-card\">"
            f"<div class=\"repo-row\">"
            f"<div class=\"repo-icon\">"
            f"<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.5\" style=\"width:18px;height:18px;color:inherit\">"
            f"<path d=\"M12 2C6.48 2 2 6.48 2 12c0 4.42 2.87 8.17 6.84 9.5.5.09.66-.22.66-.48 0-.24-.01-.87-.01-1.71-2.78.61-3.37-1.34-3.37-1.34-.45-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.61.07-.61 1.01.07 1.54 1.04 1.54 1.04.9 1.54 2.36 1.09 2.94.83.09-.65.35-1.09.63-1.34-2.22-.25-4.56-1.11-4.56-4.95 0-1.09.39-1.98 1.03-2.68-.1-.25-.45-1.27.1-2.65 0 0 .84-.27 2.75 1.02A9.56 9.56 0 0112 6.8c.85.004 1.71.115 2.51.338 1.91-1.29 2.75-1.02 2.75-1.02.55 1.38.2 2.4.1 2.65.64.7 1.03 1.59 1.03 2.68 0 3.85-2.34 4.7-4.57 4.95.36.31.68.92.68 1.85 0 1.34-.01 2.42-.01 2.75 0 .27.16.58.67.48A10.02 10.02 0 0022 12c0-5.52-4.48-10-10-10z\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/></svg>"
            f"</div>"
            f"<div class=\"repo-body\">"
            f"<div class=\"repo-title\"><a href=\"{repo_url}\" target=\"_blank\" style=\"color:inherit;text-decoration:none\">{repo_display}</a></div>"
            f"<div class=\"repo-desc\">View on GitHub</div>"
            f"</div>"
            f"</div>"
            f"</div>"
        )
        st.sidebar.markdown(repo_card, unsafe_allow_html=True)

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
    st.title("AI Speech Transcription Pipeline")


    # Author block displayed in the sidebar (professional attribution)
    _render_author_blocks(
        author_name="Shahzaib Asif",
        author_url="https://github.com/shahzaibasif",
        author_email="shahzaib.asif024@gmail.com",
        website="https://shahzaibasif.github.io",
        repo_url="https://github.com/shahzaibasif/transcription-pipeline-task",
    )

    st.write("Upload an audio file or record directly from your microphone. The app will preprocess and transcribe it.")
    st.caption("Author and repo info are in the sidebar.")

    source_choice = st.radio("Audio source", ["Upload file", "Record microphone"], horizontal=True)

    uploaded = None
    recorded = None

    if source_choice == "Upload file":
        uploaded = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])
    else:
        recorded = st.audio_input("Record audio", key="mic_audio_input")

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

    audio_source = uploaded if uploaded is not None else recorded
    if audio_source is not None:
        file_name = getattr(audio_source, "name", "microphone.wav")
        file_ext = os.path.splitext(file_name)[1].lower() or ".wav"
        if file_ext not in (".wav", ".mp3", ".m4a"):
            st.error("Unsupported file type. Please upload WAV, MP3, or M4A, or record audio from the microphone.")
            return

        tmp_dir = tempfile.gettempdir()
        tmp_path = os.path.join(tmp_dir, f"streamlit_source{file_ext}")
        with open(tmp_path, "wb") as f:
            f.write(audio_source.read())

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
