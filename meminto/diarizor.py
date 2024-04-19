import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from decorators import log_time


class Diarizor():
    def __init__(self, model: str, hugging_face_token: str):
        self.model = model
        self.hugging_face_token = hugging_face_token
    
    @log_time
    def diarize_audio(self, audio_source: Path) -> Annotation:
        pipeline = Pipeline.from_pretrained(
            self.model, use_auth_token=self.hugging_face_token
        )
        diarization: Annotation = pipeline(audio_source)
        print(diarization)
        return diarization