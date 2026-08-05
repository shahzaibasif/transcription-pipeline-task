import os
from faster_whisper import WhisperModel
from pipeline.preprocess import preprocess_audio

SUPPORTED_FORMATS = {".wav", ".mp3"}

# Load the Whisper model
model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8",
    download_root="models"
)

def transcribe_audio(audio_path):
    """
    Transcribe an audio file into text.
    Args:
        audio_path (str): Path to the audio file.
    Returns:
        str: Transcribed text.
    """
    segments, info = model.transcribe(audio_path, multilingual=False, language="en")

    transcript = []

    for segment in segments:
        transcript.append(segment.text.strip())

    return " ".join(transcript)


if __name__ == "__main__":
    
    audio_file = "audio/sample.mp3"
    extension = os.path.splitext(audio_file)[1].lower()

    if extension not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported audio format: {extension}. Supported formats: {SUPPORTED_FORMATS}")

    processed=preprocess_audio(audio_file)
    text = transcribe_audio(processed)

    print("\nTranscription:\n")
    print(text)