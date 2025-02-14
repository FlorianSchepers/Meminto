from pathlib import Path
from pyannote.audio import Pipeline
from pyannote.core import Annotation
import torch
from meminto.decorators import log_time
import whisperx
from whisperx.types import TranscriptionResult

from meminto.transcriber import TranscriptSection
from meminto.whisperxtranscriber import Transcript

class Diarizer:

    def diarize_audio(self, audio_source: Path) -> Annotation:
        raise NotImplementedError("Not implemented")

    def diarization_to_text(self, diarization: Annotation) -> str:
        diarization_text = ""
        for speech_turn, _track, speaker in diarization.itertracks(yield_label=True):
            diarization_text += (
                f"{speech_turn.start:4.1f} {speech_turn.end:4.1f} {speaker}\n"
            )
        return diarization_text

class PyannoteDiarizer:
    def __init__(self, model: str, hugging_face_token: str):
        self.pipeline = Pipeline.from_pretrained(
            model, use_auth_token=hugging_face_token
        )
        if torch.cuda.is_available():
            self.pipeline.to(torch.device("cuda"))
            print("Running pyannote.audio on GPU")
        else:
            print("Running pyannote.audio on CPU")

    @log_time
    def diarize_audio(self, audio_source: Path) -> Annotation:
        diarization = self.pipeline(audio_source)
        assert isinstance(diarization, Annotation)
        return diarization


class WhisperXDiarizer:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"Running WhisperX diarization on GPU")
        else:
            self.device = "cpu"
            print(f"Running WhisperX diarization on CPU")

        self.diarize_model = whisperx.DiarizationPipeline(
            device=self.device, use_auth_token=None
        )

    @log_time
    def diarize_audio(self, audio_file: Path, transcription_results: TranscriptionResult) -> Transcript:
        diarization_results = self.diarize_model(
            audio_file, min_speakers=2, max_speakers=2
        )

        # Map the segments to the speakers
        labled_transcription = whisperx.assign_word_speakers(
            diarization_results, transcription_results
        )
        
        transcript_sections = []
        for segment in labled_transcription["segments"]:
            transcript_section = TranscriptSection(
                start=segment["start"],
                end=segment["end"],
                speaker=segment["speaker"],
                text=segment["text"],
            )
            transcript_sections.append(transcript_section)

        return Transcript(
            text=" ".join([section.text for section in transcript_sections]),
            sections=transcript_sections,
        )