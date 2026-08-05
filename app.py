import os
from pydub import AudioSegment

from pipeline.preprocess import preprocess_audio
from pipeline.transcriber import transcribe
from pipeline.exporter import save_output

LANGUAGE = "en"  # Default language for transcription

SUPPORTED_FORMATS = {".wav", ".mp3"}

# Audio longer than this threshold will be chunked
LONG_AUDIO_THRESHOLD_MINUTES = 2
CHUNK_DURATION_MINUTES = 1

def process_normal_audio(audio_path):

    processed = preprocess_audio(audio_path)
    result = transcribe(processed, multilingual=False, language=LANGUAGE)

    if os.path.exists(processed):
        os.remove(processed)

    return result

# Large audio handling: split into chunks, transcribe each, and merge results
def process_long_audio(audio_path):
    """
    Split long audio into chunks, transcribe each chunk,
    adjust timestamps, and merge the results.
    """

    audio = AudioSegment.from_file(audio_path)

    chunk_length_ms = CHUNK_DURATION_MINUTES * 60 * 1000

    all_segments = []
    full_text = []

    total_chunks = (len(audio) + chunk_length_ms - 1) // chunk_length_ms

    print(f"\nLong audio detected.")
    print(f"Splitting into {total_chunks} chunks...\n")

    for i in range(total_chunks):

        start_ms = i * chunk_length_ms
        end_ms = min((i + 1) * chunk_length_ms, len(audio))

        chunk = audio[start_ms:end_ms]

        chunk_file = f"temp_chunk_{i}.wav"
        chunk.export(chunk_file, format="wav")

        processed = preprocess_audio(chunk_file)

        result = transcribe(processed, multilingual=False, language=LANGUAGE)

        offset = start_ms / 1000.0

        for segment in result["segments"]:
            segment["start"] += offset
            segment["end"] += offset
            all_segments.append(segment)

        full_text.append(result["text"])

        os.remove(chunk_file)

        if os.path.exists(processed):
            os.remove(processed)

        print(f"Finished chunk {i + 1}/{total_chunks}")

    return {
        "language": result["language"],
        "text": " ".join(full_text),
        "segments": all_segments
    }


def main():

    input_audio = "audio/sample.mp3"
    extension = os.path.splitext(input_audio)[1].lower()

    if extension not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported audio format: {extension}. Supported formats: {SUPPORTED_FORMATS}")

    audio = AudioSegment.from_file(input_audio)

    duration_seconds = len(audio) / 1000
    duration_minutes = len(audio) / (1000 * 60)
    
    print(f"\nAudio duration: {duration_seconds:.2f} seconds ({duration_minutes:.2f} minutes)")

    if duration_minutes >= LONG_AUDIO_THRESHOLD_MINUTES:
        result = process_long_audio(input_audio)
    else:
        result = process_normal_audio(input_audio)

    save_output(result)

    print("\n========== FINAL TRANSCRIPT ==========\n")
    print(result["text"])


if __name__ == "__main__":
    main()