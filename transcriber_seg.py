import json, os
from faster_whisper import WhisperModel
from pipeline.preprocess import preprocess_audio

SUPPORTED_FORMATS = {".wav", ".mp3"}

# Load Whisper model (runs on CPU; use device="cuda" for GPU)
model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8",
    download_root="models"
)

def transcribe_audio(audio_path):
    """
    Transcribe an audio file and return text with timestamps.

    Args:
        audio_path (str): Path to audio file.

    Returns:
        dict: Transcription result with timestamps.
    """
    segments, info = model.transcribe(audio_path, multilingual=False, language="en")

    transcript = []
    segment_list = []

    for segment in segments:
        transcript.append(segment.text.strip())

        segment_list.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip()
        })

    return {
        "language": info.language,
        "text": " ".join(transcript),
        "segments": segment_list
    }


if __name__ == "__main__":

    audio_file = "audio/sample2.mp3"
    extension = os.path.splitext(audio_file)[1].lower()

    if extension not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported audio format: {extension}. Supported formats: {SUPPORTED_FORMATS}")

    processed=preprocess_audio(audio_file)
    result = transcribe_audio(processed)

    # Print complete transcript
    print("\nFull Transcript:\n")
    print(result["text"])

    print("\nSegments:\n")

    for segment in result["segments"]:
        print(
            f"[{segment['start']:.2f}s - {segment['end']:.2f}s] "
            f"{segment['text']}"
        )

    # Save JSON
    with open("transcript.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)