from faster_whisper import WhisperModel

model=WhisperModel("base",
                   device="cpu",
                   compute_type="int8", 
                   download_root="models"
                   )

def transcribe(audio_file, multilingual=False, language=None):
    segments,info=model.transcribe(audio_file, multilingual=multilingual, language=language)
    res=[]; txt=[]
    for s in segments:
        txt.append(s.text)
        res.append({"start":s.start,"end":s.end,"text":s.text.strip()})
    return {"language":info.language,"duration":res[-1]["end"] if res else 0,"text":" ".join(txt).strip(),"segments":res}
