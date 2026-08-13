import os
import tempfile
import streamlit as st
import pandas as pd

from pipeline.preprocess import preprocess_audio
from pipeline.transcriber import transcribe
from pipeline.exporter import save_output


st.set_page_config(page_title="Transcription Pipeline UI", layout="centered")


def transcribe_file(input_path: str, multilingual: bool, language: str):
    processed = preprocess_audio(input_path)
    result = transcribe(processed, multilingual=multilingual, language=language)
    try:
        if os.path.exists(processed):
            os.remove(processed)
    except Exception:
        pass
    return result


def main():
    st.title("Transcription Pipeline")

    st.write("Upload an audio file (WAV or MP3). The app will preprocess and transcribe it using the project's pipeline.")

    uploaded = st.file_uploader("Upload audio", type=["wav", "mp3"])

    col1, col2 = st.columns(2)
    with col1:
        multilingual = st.checkbox("Multilingual", value=False)
    with col2:
        language = st.text_input("Language (ISO code)", value="en")

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
                    result = transcribe_file(tmp_path, multilingual=multilingual, language=language)
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
