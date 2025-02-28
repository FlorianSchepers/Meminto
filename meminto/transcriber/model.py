from dataclasses import dataclass, field
from typing import List
from whisperx.types import AlignedTranscriptionResult, TranscriptionResult, SingleSegment

@dataclass
class TranscriptSection:
    start: float
    end: float
    text: str
    speaker: str | None = None

    def __str__(self):
        speaker_str = self.speaker if self.speaker is not None else "Unknown Speaker"
        return f"[{self.start:.2f} - {self.end:.2f}] {speaker_str}: {self.text}"

@dataclass
class Transcript:
    text: str
    language: str | None = None
    sections: List[TranscriptSection] = field(default_factory=list)

    def __str__(self):
        return "\n".join(str(chunk) for chunk in self.sections)