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

## Conclusion

This project demonstrates a complete engineering solution for a speech-to-text pipeline. The system is modular, scalable, and designed with production best practices in mind. By separating preprocessing, transcription, post-processing, and export into independent components, the architecture remains maintainable and extensible. The use of asynchronous processing, structured storage, robust error handling, and a REST API enables the solution to scale from a simple command-line utility to a production-grade transcription service.

## Author
Shahzaib Asif @ [Email](mailto:shahzaib.asif024@gmail.com)