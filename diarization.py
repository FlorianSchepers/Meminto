import os
import pickle
from pyannote.audio import Pipeline

from decorators import log_time

HUGGING_FACE_ACCESS_TOKEN = os.environ["HUGGING_FACE_ACCESS_TOKEN"]

def load_diarization(file_path):
    with open(file_path, 'rb') as file:
        diarization = pickle.load(file)
    return diarization

def save_diarization(diarization, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(diarization, file, pickle.HIGHEST_PROTOCOL)


@log_time
def diarize_audio(audio_source):
    print("Diarizing audio")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=HUGGING_FACE_ACCESS_TOKEN)
    return pipeline(audio_source.resolve())