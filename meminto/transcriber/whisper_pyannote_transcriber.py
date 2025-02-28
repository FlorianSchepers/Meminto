import logging
from accelerate import Accelerator
from pathlib import Path
import torch
from meminto.audio_processing import load_audio
import whisperx
from whisperx.types import TranscriptionResult, SingleSegment

import pandas as pd

from pyannote.audio import Pipeline
from pyannote.core import Annotation

from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    pipeline,
)
from meminto.logging_config import log_time
from meminto.transcriber.model import Transcript, TranscriptSection
from meminto.transcriber.base_transcriber import BaseTranscriber
from meminto.transcriber.supported_models import PYANNOTE_AUDIO_MODELS, WHISPER_MODELS

class WhisperPyannoteTranscriber(BaseTranscriber):
    def __init__(
        self,
        hugging_face_token: str,
        whisper_model_name: WHISPER_MODELS = WHISPER_MODELS.LARGE,
        pyannote_model_name: PYANNOTE_AUDIO_MODELS = PYANNOTE_AUDIO_MODELS.SPEAKER_DIARIZATION_2_1 
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.pyannote_pipeline = Pipeline.from_pretrained(
            pyannote_model_name.value, use_auth_token=hugging_face_token
            )

        self.accelerator = Accelerator()
        
        if self.accelerator.device.type == "cuda":
            self.pyannote_pipeline.to(torch.device("cuda"))
            device_map = "auto"
            torch_dtype = torch.float16
            self.logger.info(f"Running Whisper and pyannote.audio on GPU with dtype {torch_dtype}")
        else:
            device_map = None
            torch_dtype = torch.float32
            self.logger.info(f"Running Whisper and pyannote.audio on CPU with dtype {torch_dtype}")
        
        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            whisper_model_name.value,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            device_map=device_map,
        )

        whisper_model = self.accelerator.prepare(whisper_model)
        whisper_processor = AutoProcessor.from_pretrained(whisper_model_name.value)
        self.whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model=whisper_model,
            tokenizer=whisper_processor.tokenizer,
            feature_extractor=whisper_processor.feature_extractor,
            torch_dtype=torch_dtype,
            chunk_length_s=30,
            stride_length_s=5,
        )

    @log_time
    def create_transcript(self, audio_file: Path) -> Transcript:
        transcription_result = self.transcribe(audio_file)
        diarization_result = self.diarize(audio_file)
        transcript = self.assigne_speakers(diarization_result, transcription_result)

        return transcript


    @log_time
    def transcribe(self, audio_file: Path) -> Transcript:
        self.logger.info("Started Transcription")
        audio = load_audio(audio_file).numpy()
        transcript = self.whisper_pipeline(audio, return_timestamps=True)

        transcript_chunks = transcript.get("chunks", [])

        transcript_sections = [
            TranscriptSection(
                start=chunk["timestamp"][0],
                end=chunk["timestamp"][1],
                text=chunk["text"],
            )
            for chunk in transcript_chunks
        ]

        return Transcript(
            text=transcript.get("text", ""),
            sections=transcript_sections,
        )
    
    @log_time
    def diarize(self, audio_file: Path) -> Annotation:
        diarization = self.pyannote_pipeline(audio_file)
        assert isinstance(diarization, Annotation)
        return diarization
    
    @log_time
    def assigne_speakers(self, diarization: Annotation, transcription: Transcript) -> Transcript:
        data = []
        for segment, _track, label in diarization.itertracks(yield_label=True):
            start = segment.start
            end = segment.end
            speaker = label
            data.append({'start': start, 'end': end, 'speaker': speaker})   
        diarization_df = pd.DataFrame(data)

        segments = [
            SingleSegment(start=section.start, end=section.end, text=section.text)
            for section in transcription.sections
        ]
        transcription_result = TranscriptionResult(segments=segments, language=transcription.language or "")
        
        
        labled_transcript = whisperx.assign_word_speakers(diarization_df, transcription_result)
        
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