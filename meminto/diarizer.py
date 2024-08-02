from pathlib import Path
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from meminto.decorators import log_time


class Diarizer:
    def __init__(self, model: str, hugging_face_token: str):
        self.pipeline = Pipeline.from_pretrained(
            model, use_auth_token=hugging_face_token
        )

    @log_time
    def diarize_audio(self, audio_source: Path) -> Annotation:
        diarization = self.pipeline(audio_source)
        assert isinstance(diarization, Annotation)
        return diarization

    def diarization_to_text(self, diarization: Annotation) -> str:
        diarization_text = ""
        for speech_turn, _track, speaker in diarization.itertracks(yield_label=True):
            diarization_text += (
                f"{speech_turn.start:4.1f} {speech_turn.end:4.1f} {speaker}\n"
            )
        return diarization_text
