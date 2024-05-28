from pathlib import Path
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from meminto.decorators import log_time


class Diarizer:
    def __init__(self, model: str, hugging_face_token: str):
        self.model = model
        self.hugging_face_token = hugging_face_token

    @log_time
    def diarize_audio(self, audio_source: Path) -> Annotation:
        pipeline = Pipeline.from_pretrained(
            self.model, use_auth_token=self.hugging_face_token
        )
        diarization = pipeline(audio_source)
        assert isinstance(diarization, Annotation)
        return diarization

