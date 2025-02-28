
import logging
from pathlib import Path
from pandas import DataFrame
import torch
import whisperx
from whisperx.types import AlignedTranscriptionResult, TranscriptionResult, SingleSegment
from meminto.logging_config import log_time
from meminto.transcriber.model import Transcript, TranscriptSection
from meminto.transcriber.base_transcriber import BaseTranscriber
from meminto.transcriber.supported_models import WHISPER_X_MODELS


class WhisperXTranscriber(BaseTranscriber):
    def __init__(
        self,
        whisper_x_model_name: WHISPER_X_MODELS = WHISPER_X_MODELS.LARGE_V2,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        if torch.cuda.is_available():
            self.device = "cuda"
            compute_type = "float16"
            self.logger.info(f"Running WhisperX on GPU with dtype {compute_type}")
        else:
            self.device = "cpu"
            compute_type = torch.float32
            self.logger.info(f"Running WhisperX on CPU with dtype {compute_type}")

        self.transcription_model = whisperx.load_model(
            whisper_x_model_name.value, device=self.device, compute_type=compute_type
        )

        self.diarization_model = whisperx.DiarizationPipeline(
            device=self.device, use_auth_token=None
        )

    @log_time
    def create_transcript(self, audio_file: Path) -> Transcript:
        transcription_result = self.transcribe(audio_file)
        diarization_result = self.diarize(audio_file)
        transcript = self.assigne_speakers(diarization_result, transcription_result)

        return transcript

    @log_time
    def transcribe(self, audio_file: Path) -> AlignedTranscriptionResult:
        self.logger.info("Started Transcription")
        audio = whisperx.load_audio(audio_file.as_posix())
        transcript = self.transcription_model.transcribe(audio, batch_size=16)
    
        language_code = transcript["language"]
        alginment_model, metadata = whisperx.load_align_model(language_code=language_code, device=self.device)
        return whisperx.align(transcript["segments"], alginment_model, metadata, audio, self.device, return_char_alignments=False)

    @log_time
    def diarize(self, audio_file: Path) -> DataFrame:
        self.logger.info("Started Diarization")
        return  self.diarization_model(
            audio_file.as_posix()
        )
    
    @log_time
    def assigne_speakers(self, diarization_result: DataFrame, transcription_result: AlignedTranscriptionResult) -> Transcript:
        self.logger.info("Started speaker assignment")
        labled_transcript = whisperx.assign_word_speakers(
            diarization_result, transcription_result
        )

        transcript_sections = []
        for segment in labled_transcript["segments"]:
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