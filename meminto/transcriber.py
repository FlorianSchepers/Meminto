from dataclasses import dataclass
from enum import Enum
from io import BytesIO
import requests
import torch
import torchaudio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline,
)
from meminto.decorators import log_time
from meminto.audio_processing import SAMPLING_RATE, AudioSection


class WHISPER_MODEL_SIZE(Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    DISTIL = "distil"
    LARGE = "large"


@dataclass
class TranscriptSection:
    start: float
    end: float
    speaker: str
    text: str

    def __str__(self):
        return f'{self.speaker}: "{self.text}"\n'


class Transcriber:
    def transcribe_section(self, audio_section: AudioSection):
        raise NotImplementedError("Subclasses should implement this method")

    @log_time
    def transcribe(self, audio_sections: list[AudioSection]) -> list[TranscriptSection]:
        transcript_sections = []
        total_number_of_sections = len(audio_sections)
        for section_number, section in enumerate(audio_sections):
            print(
                f"Transscribing section {section_number} of {total_number_of_sections}."
            )
            transcription = self.transcribe_section(section)
            transcript_section = TranscriptSection(
                start=section.turn.start,
                end=section.turn.end,
                speaker=section.speaker,
                text=transcription,
            )
            transcript_sections.append(transcript_section)
        return transcript_sections

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


class LocalTranscriber(Transcriber):
    def __init__(
        self,
        model_size: WHISPER_MODEL_SIZE = WHISPER_MODEL_SIZE.MEDIUM,
        english_only: bool = False,
    ):
        model_name = _model_name(model_size, english_only)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Running on {device}")
        whisper_processor = WhisperProcessor.from_pretrained(model_name)
        whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name).to(
            device
        )
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=whisper_model,
            tokenizer=whisper_processor.tokenizer,
            feature_extractor=whisper_processor.feature_extractor,
            chunk_length_s=30,
            stride_length_s=5,
            batch_size=8,
            max_new_tokens=128,
            device=device,
            torch_dtype=torch_dtype,
        )

    def transcribe_section(self, audio_section: AudioSection):
        section_transcript = self.pipeline(
            audio_section.audio.numpy(),
        )
        return section_transcript["text"].strip()

    # @log_time
    # def transcribe(self, audio_sections: list[AudioSection]) -> list[TranscriptSection]:
    #     transcript_sections = []
    #     total_number_of_sections = len(audio_sections)
    #     for section_number, section in enumerate(audio_sections):
    #         print(
    #             f"Transscribing section {section_number} of {total_number_of_sections}."
    #         )
    #         transcription = self.pipeline(
    #             section.audio.numpy(),
    #         )
    #         transcript_section = TranscriptSection(
    #             start=section.turn.start,
    #             end=section.turn.end,
    #             speaker=section.speaker,
    #             text=transcription["text"].strip(),
    #         )
    #         transcript_sections.append(transcript_section)
    #     return transcript_sections


class RemoteTranscriber(Transcriber):
    def __init__(
        self,
        url: str,
        authorization: str,
    ):
        self.url = url
        self.authorization = authorization

    @log_time
    def transcribe_section(
        self, audio_section: AudioSection
    ) -> list[TranscriptSection]:
        buffer = BytesIO()
        torchaudio.save(
            buffer,
            audio_section.audio.unsqueeze(0),
            format="wav",
            sample_rate=SAMPLING_RATE,
        )
        buffer.seek(0)

        headers = {
            "Authorization": self.authorization,
        }
        files = {"file": buffer, "response_format": (None, "json")}
        print(f"Endpoint used for transcription: {self.url}")
        response = requests.post(url=self.url, headers=headers, files=files)
        section_transcript = response.json()
        return section_transcript["text"].strip()
    
    def diarize_audio(
            self, audio_path: str
            ) -> str:
        
        headers = {
            "Authorization": self.authorization,
        }
        files = {"file": open(audio_path, "rb"), "response_format": (None, "verbose_json")}
        print(f"Endpoint used for transcription: {self.url}")
        response = requests.post(url=self.url, headers=headers, files=files)
        print(response)
        print(response.text)
        return response.json()

def _model_name(model_size: WHISPER_MODEL_SIZE, english_only: bool) -> str:
    match model_size:
        case WHISPER_MODEL_SIZE.TINY:
            if english_only:
                whisper_model_name = "openai/whisper-tiny.en"  # English-only, ~ 151 MB
            else:
                whisper_model_name = "openai/whisper-tiny"  # multilingual, ~ 151 MB
        case WHISPER_MODEL_SIZE.BASE:
            if english_only:
                whisper_model_name = "openai/whisper-base.en"  # English-only, ~ 290 MB
            else:
                whisper_model_name = "openai/whisper-base"  # multilingual, ~ 290 MB
        case WHISPER_MODEL_SIZE.SMALL:
            if english_only:
                whisper_model_name = "openai/whisper-small.en"  # English-only, ~ 967 MB
            else:
                whisper_model_name = "openai/whisper-small"  # multilingual, ~ 967 MB
        case WHISPER_MODEL_SIZE.MEDIUM:
            if english_only:
                whisper_model_name = (
                    "openai/whisper-medium.en"  # English-only, ~ 3.06 GB
                )
            else:
                whisper_model_name = "openai/whisper-medium"  # multilingual, ~ 3.06 GB
        case WHISPER_MODEL_SIZE.DISTIL:
            whisper_model_name = "distil-whisper/distil-large-v2"
        case WHISPER_MODEL_SIZE.LARGE:
            whisper_model_name = "openai/whisper-large-v2"  # multilingual, ~ 6.17 GB
        case _:
            whisper_model_name = "openai/whisper-medium"  # multilingual, ~ 3.06 GB
    return whisper_model_name
