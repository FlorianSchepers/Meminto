import pickle
from pyannote.audio import Pipeline

def load_diarization(file_path):
    with open(file_path, 'rb') as file:
        diarization = pickle.load(file)
    return diarization

def save_diarization(diarization, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(diarization, file, pickle.HIGHEST_PROTOCOL)

def diarize_audio(audio_source):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")
    return pipeline(audio_source)