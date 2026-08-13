import os
import tempfile
import streamlit as st
import pandas as pd

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
            with st.spinner("Transcribing — this may take a while while the model runs..."):
                try:
                    # Pass selected_model into transcribe_file; if faster-whisper is not available
                    # pipeline_transcribe will be used instead.
                    result = transcribe_file(tmp_path, multilingual=multilingual, language=(None if language == "auto" else language), model_name=selected_model)
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    return

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
