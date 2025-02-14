from dataclasses import dataclass, field
from accelerate import Accelerator
from pathlib import Path
from typing import List, Tuple
import torch
from meminto.decorators import log_time
from meminto.audio_processing import AudioSection, load_audio
import whisperx
from enum import Enum

from pyannote.audio import Pipeline
from pyannote.core import Annotation

from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    pipeline,
)
from meminto.transcriber import Transcriber, TranscriptSection


class WHISPER_MODELS(Enum):
    TINY_EN = "openai/whisper-tiny.en"
    TINY = "openai/whisper-tiny"
    BASE_EN = "openai/whisper-base.en"
    BASE = "openai/whisper-base"
    SMALL_EN = "openai/whisper-small.en"
    SMALL = "openai/whisper-small"
    MEDIUM_EN = "openai/whisper-medium.en"
    MEDIUM = "openai/whisper-medium"
    LARGE = "openai/whisper-large-v2"
    DISTIL_LARGE = "distil-whisper/distil-large-v2"


class WHISPERX_MODELS(Enum):
    LARGE_V2 = "large-v2"


class PYANNOTE_AUDIO_MODELS(Enum):
    SPEAKER_DIARIZATION_2_1 = "pyannote/speaker-diarization@2.1"


@dataclass
class Transcript:
    text: str
    sections: List[TranscriptSection] = field(default_factory=list)
    language: str | None

    def __str__(self):
        return "\n".join(str(chunk) for chunk in self.sections)


class TranscriberEmbbed:
    def transcribe(self, audio_file: Path) -> list[TranscriptSection]:
        raise NotImplementedError("Subclasses should implement this method")

    def transcript_to_txt(self, transcript: list[TranscriptSection]):
        transcript_text = ""
        for transcript_section in transcript:
            transcript_text += (
                f"start={transcript_section.start:.1f}s "
                + f"end={transcript_section.end:.1f}s "
                + f"speaker={transcript_section.speaker}:\n"
            )
            transcript_text += transcript_section.text + "\n"
        return transcript_text


class WhisperTranscriber(TranscriberEmbbed):
    def __init__(
        self,
        whisper_model_name: WHISPER_MODELS = WHISPER_MODELS.LARGE,
    ):
        self.accelerator = Accelerator()

        if self.accelerator.device.type == "cuda":
            device_map = "auto"
            torch_dtype = torch.float16
            print(f"Running Whisper on GPU with dtype {torch_dtype}")
        else:
            device_map = None
            torch_dtype = torch.float32
            print(f"Running Whisper on CPU with dtype {torch_dtype}")

        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            whisper_model_name.value,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            device_map=device_map,
        )

        whisper_model = self.accelerator.prepare(whisper_model)

        processor = AutoProcessor.from_pretrained(whisper_model_name.value)

        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=whisper_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            chunk_length_s=30,
            stride_length_s=5,
        )

    @log_time
    def transcribe(self, audio_file: Path) -> list[TranscriptSection]:
        audio_tensor = load_audio(audio_file)
        audio_input = audio_tensor.numpy()

        transcription_result = self.pipeline(audio_input, return_timestamps=True)

        transcript_chunks = transcription_result.get("chunks", [])

        transcript_sections = [
            TranscriptSection(
                start=chunk["timestamp"][0],
                end=chunk["timestamp"][1],
                text=chunk["text"],
            )
            for chunk in transcript_chunks
        ]

        return Transcript(
            text=transcription_result.get("text", ""),
            sections=transcript_sections,
        )


class WhisperXTranscriber(TranscriberEmbbed):
    def __init__(
        self,
        whisper_x_model_name: WHISPERX_MODELS = WHISPERX_MODELS.LARGE_V2,
    ):
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
            print(f"Running WhisperX on GPU with dtype {compute_type}")
        else:
            device = "cpu"
            compute_type = torch.float32
            print(f"Running WhisperX on CPU with dtype {compute_type}")

        self.model = whisperx.load_model(
            whisper_x_model_name.value, device=device, compute_type=compute_type
        )

    @log_time
    def transcribe(self, audio_file: Path) -> List[TranscriptSection]:
        audio = whisperx.load_audio(audio_file.as_posix())

        transcription_result = self.model.transcribe(audio, batch_size=16)

        transcript_sections = [
            TranscriptSection(
                start=chunk["start"],
                end=chunk["end"],
                text=chunk["text"],
            )
            for chunk in transcription_result["segments"]
        ]

        return Transcript(
            text=transcription_result.get("text", ""),
            sections=transcript_sections,
        )


class WhisperXTranscriberV1(Transcriber):
    def __init__(
        self,
        whisper_x_model_name: WHISPERX_MODELS = WHISPERX_MODELS.LARGE_V2,
        compute_type: str = "float32",
        device: str = "cpu",
    ):
        if device == "gpu" and not torch.cuda.is_available():
            raise ValueError("GPU is not available")
        if device == "cpu" and compute_type == "float16":
            raise ValueError("Cannot use float16 on CPU")

        self.device = device

        self.transcription_model = whisperx.load_model(
            whisper_x_model_name, device=self.device, compute_type=compute_type
        )

        self.diarize_model = whisperx.DiarizationPipeline(
            device=self.device, use_auth_token=None
        )

    @log_time
    def transcribe(self, audio_file: Path) -> List[TranscriptSection]:
        audio = whisperx.load_audio(audio_file)

        transcription_result = self.transcription_model.transcribe(audio, batch_size=16)

        diarize_segments = self.diarize_model(
            audio_file, min_speakers=2, max_speakers=2
        )
        

        segment_speaker_mapping = whisperx.assign_word_speakers(
            diarize_segments, transcription_result
        )

        transcript_sections = []
        for segment in segment_speaker_mapping["segments"]:
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
