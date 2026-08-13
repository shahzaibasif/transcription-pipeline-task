# Speech-to-Text Transcription Pipeline
(Software Engineer Task @ Shahzaib Asif)
## Overview

This project implements a modular speech-to-text (STT) pipeline that converts spoken language from an audio file into text while preserving timestamps for each transcription segment.

The implementation focuses on engineering design rather than training a speech recognition model. The pipeline uses **Faster-Whisper**, an optimized implementation of OpenAI's Whisper model, to provide accurate and efficient transcription.

The solution is designed to be modular, scalable, and production-ready, making it suitable for integration into larger AI or data processing systems.

---

# Objectives

The pipeline provides the following functionality:

- Accept audio files (WAV, MP3) later support for M4A, FLAC, OGG, and AAC can be added.
- Automatically preprocess audio into a standard format
- Convert speech into text
- Return timestamps for every transcription segment
- Export results as plain text and JSON
- Support downstream NLP or LLM applications
- Be easily deployable as an API

---

# Technology Stack

| Component | Technology |
|------------|------------|
| Language | Python 3.12 |
| Speech-to-Text | Faster-Whisper |
| Audio Processing | Pydub + FFmpeg |

---

# Project Structure

```
Transcription Pipeline/

│
├── app.py
├── requirements.txt
├── README.md
│
├── audio/  
│   └── sample.mp3
│
├── output/
│   ├── transcript.txt
│   └── transcript.json
│
├── pipeline/
│   ├── preprocess.py
│   ├── transcriber.py
│   └── exporter.py
│
├── models/
|    └── (model files)
```

---

# Pipeline Architecture

```
                Audio File
                    │
                    ▼
        Audio Validation
                    │
                    ▼
        Audio Preprocessing
     (Mono + 16kHz + WAV)
                    │
                    ▼
      Faster-Whisper Model
                    │
                    ▼
        Speech Segments
                    │
                    ▼
        Post Processing
                    │
                    ▼
      Transcript + Timestamps
                    │
                    ▼
     JSON / Text Export / API
```

---

# Pipeline Components

## 1. Audio Input

The system accepts audio files in the following formats:

- WAV
- MP3

`later support for M4A, FLAC, OGG, and AAC can be added.`

---

## 2. Audio Validation

Before transcription the pipeline validates:

- File exists
- Supported format
- Audio duration
- Audio integrity

If validation fails an appropriate error is returned.

---

## 3. Audio Preprocessing

Different recording devices produce different sample rates and channel configurations.

To ensure consistent transcription quality every input file is converted into

- WAV
- 16 kHz
- Mono

This preprocessing is performed using Pydub with FFmpeg.

Benefits:

- Better transcription accuracy
- Consistent input format
- Reduced model errors

---

## 4. Speech-to-Text

The project uses Faster-Whisper.

Reasons:

- Open source
- High accuracy
- Faster than original Whisper
- CPU and GPU support
- Automatic language detection
- Timestamp generation

The model returns:

- text
- start timestamp
- end timestamp

---

## 5. Post Processing

The transcript is cleaned by

- removing unnecessary spaces
- merging segments
- restoring punctuation
- formatting output

---

## 6. Export

Two outputs are generated. (Plain Text and JSON files)

### Plain Text

```
Hello everyone. Welcome to today's meeting. We will discuss the project roadmap.
```

### JSON (example)

```json
{
    "language": "en",
    "duration": 5.9,
    "text": "Hello everyone. Welcome to today's meeting. We will discuss the project roadmap.",
    "segments": [
        {
            "start": 0.0,
            "end": 1.2,
            "text": "Hello everyone."
        },
        {
            "start": 1.2,
            "end": 3.6,
            "text": "Welcome to today's meeting."
        },
        {
            "start": 3.6,
            "end": 5.9,
            "text": "We will discuss the project roadmap."
        }
    ]
}
```

---

# Engineering Design Decisions

## Why Faster-Whisper?

Faster-Whisper provides nearly the same transcription accuracy as Whisper while offering significantly faster inference and lower memory consumption. It supports both CPU and GPU execution, making it suitable for development and production environments.

---

## How do you handle different audio formats?

The pipeline accepts common audio formats such as WAV, MP3, M4A, FLAC, AAC, and OGG.

Regardless of the original format, every file is converted to a standard format before transcription:

- WAV
- 16 kHz
- Mono

Using a standardized format improves transcription quality and avoids inconsistencies caused by different sampling rates or stereo recordings.

---

## How do you deal with long audio files?

Long recordings are processed using **chunking**.

Instead of transcribing an entire recording at once:

1. Split the audio into 60 second chunks.
2. Transcribe each chunk independently.
3. Adjust timestamps using the chunk offset.
4. Merge all segments into a final transcript.

Advantages:

- Lower memory usage
- Better fault tolerance
- Parallel processing
- Faster overall execution

```
                Audio File
                    │
                    ▼
              Split into Chunks
                    │
                    ▼
            Transcribe Each Chunk
                    │
                    ▼
            Adjust Timestamps
                    │
                    ▼
            Merge Segments  
                    │
                    ▼
            Final Transcript   

```

---

## How would you handle concurrent uploads?

For production deployments I would expose the service through FastAPI and process transcription jobs asynchronously.

Architecture:

```
        Clients
           │
           ▼
     FastAPI REST API
           │
           ▼
     Upload & Validate
           │
           ▼
      Message Queue
 (Celery + Redis/RabbitMQ)
           │
           ▼
   Multiple Worker Processes
           │
           ▼
      Whisper Transcription
           │
           ▼
   Store Results (JSON/DB)
           │
           ▼
      Client Retrieves Result
```

Workflow:

- User uploads audio.
- API immediately returns a Job ID.
- Worker processes transcription.
- Client retrieves results later.

Benefits:

- Non-blocking API
- Multiple simultaneous uploads
- Horizontal scalability

---

## How would you store audio and transcripts?

Audio files and transcripts have different storage requirements.

### Audio

Store in object storage.

Examples:

- AWS S3
- Azure Blob Storage
- Google Cloud Storage

```
For a production system, I would store audio files and transcripts separately because they have different storage and access patterns.

Client
   │
   ▼
Upload API
   │
   ├──────────────► Object Storage (Audio Files)
   │                  • AWS S3
   │                  • Azure Blob Storage
   │                  • Google Cloud Storage
   │
   ▼
Transcription Worker
   │
   ▼
Database (Metadata & Transcripts)
    • PostgreSQL
    • MongoDB
```
### Transcript

Store inside PostgreSQL or MongoDB.

Database stores:

- Job ID
- Audio URL
- Transcript
- Segments
- Status
- Language
- Processing time

This separation improves scalability.

---

## How do you retry or recover failed transcriptions?

Every transcription job has a status.

```
Queued

Processing

Completed

Failed
```

Transient failures:

- Worker crash
- Network issue
- Storage unavailable

These are retried using exponential backoff.

Example:

Attempt 1

↓

Attempt 2 (2 sec)

↓

Attempt 3 (4 sec)

↓

Failed

Permanent failures such as corrupted audio are not retried.

Architecture:

```
    Upload
    │
    ▼
    Queue Job
    │
    ▼
    Worker Starts
    │
    ├── Success ─────────► Completed
    │
    └── Failure
            │
            ▼
    Retry (Max 3 Attempts)
            │
        ┌───┴────┐
        │        │
    Success     Failed
        │        │
    Completed   Log Error & Notify User
```

---

## How would you expose this as an API?

I would expose it as a REST API using FastAPI, keeping the API lightweight while offloading transcription to background workers for scalability.
```
Client
   │
   ▼
FastAPI
   │
   ├── POST /transcribe
   ├── GET  /jobs/{job_id}
   └── GET  /jobs/{job_id}/result
           │
           ▼
    Queue (Celery/Redis)
           │
           ▼
   Whisper Worker
           │
           ▼
 Database + Object Storage
```

### Upload

```
POST /transcribe
```

Upload an audio file.

Response

```json
{
    "job_id":"12345",
    "status":"queued"
}
```

---

### Check Status

```
GET /jobs/{job_id}
```

Response

```json
{
    "status":"processing"
}
```

---

### Get Result

```
GET /jobs/{job_id}/result
```

Response

```json
{
    "text":"Hello everyone",
    "segments":[]
}
```

---

<!-- ## Security Considerations

For production deployments, implement:

- HTTPS
- JWT or API Key authentication
- File size limits
- Malware scanning
- Rate limiting
- Secure object storage
- Encryption

---

## Scalability

To support thousands of transcription requests:

- Multiple FastAPI instances
- Load balancer
- Redis message queue
- Celery worker pool
- GPU inference workers
- Object storage
- Database indexing
- Horizontal scaling

---

## Future Improvements

Possible enhancements include:

- Voice Activity Detection
- Real-time streaming transcription
- Subtitle (SRT/VTT) generation
- Multi-language translation
- Confidence scores
- Keyword extraction
- Automatic summarization using LLMs
- Sentiment analysis

---

## Assumptions

This implementation assumes:

- Audio files contain human speech.
- FFmpeg is installed.
- Whisper models are downloaded automatically if not found locally.
- Timestamp accuracy is sufficient for downstream NLP applications.
- Internet access is only required for the initial model download. -->

---

## Streamlit Frontend UI

A lightweight Streamlit frontend is included as `ui_app.py` to upload audio files, run the project's preprocessing and transcription pipeline, and download/save transcripts.

Quick steps to run the UI locally:

1. (Optional) Create and activate a virtual environment for Python 3.12.

2. Install project dependencies and Streamlit:

```bash
pip install -r requirements.txt
pip install streamlit
```

3. Ensure `ffmpeg` is installed and available on your PATH (required by `pydub`). On Windows you can download FFmpeg from https://ffmpeg.org/ and add it to PATH.

4. Start the Streamlit app from the project root:

```bash
streamlit run ui_app.py
```

5. In the browser UI, upload a WAV or MP3 file, choose language/multilingual options if needed, then click "Start Transcription". You can view segments, download the transcript, or save it to the `output/` folder.

Notes:
- The UI calls the same pipeline code used by `app.py` and expects model files under the `models/` directory.
- Transcription may take several minutes depending on your CPU and which Faster-Whisper model is used.


## Conclusion

This project demonstrates a complete engineering solution for a speech-to-text pipeline. The system is modular, scalable, and designed with production best practices in mind. By separating preprocessing, transcription, post-processing, and export into independent components, the architecture remains maintainable and extensible. The use of asynchronous processing, structured storage, robust error handling, and a REST API enables the solution to scale from a simple command-line utility to a production-grade transcription service.


## Author
Shahzaib Asif @ [Email](mailto:shahzaib.asif024@gmail.com)