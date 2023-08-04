from pyannote.audio import Pipeline

def diarize_audio(audio_source):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")
    return pipeline(audio_source)