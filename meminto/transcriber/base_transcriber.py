from pathlib import Path

from meminto.transcriber.model import Transcript


class BaseTranscriber:
    def create_transcript(self, audio_file: Path) -> Transcript:
        raise NotImplementedError("Subclasses should implement this method")