from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Tuple
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline,
    Pipeline,
)
from audio_processing import SAMPLING_RATE, batch
from decorators import log_time
from helpers import Language


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


@log_time
def create_transcript(
    audio_sections: list[Any], language: Language
) -> list[TranscriptSection]:

    print("Creating transcript from diarized audio")
    whisper = setup_whisper()
    transcript_sections = transcribe_sections(audio_sections, whisper)

    return transcript_sections


def setup_whisper(
    model_size: WHISPER_MODEL_SIZE = WHISPER_MODEL_SIZE.MEDIUM,
    english_only: bool = False,
) -> Pipeline:
    model_name = get_model_name(model_size, english_only)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    whisper_processor = WhisperProcessor.from_pretrained(model_name)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        model_name
    ).to(device)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    whisper = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=whisper_processor.tokenizer,
        feature_extractor=whisper_processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )
    return whisper

def get_model_name(model_size: WHISPER_MODEL_SIZE, english_only: bool)->str:
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

def transcribe_sections(audio_sections: list[Any], whisper: Pipeline) -> list[TranscriptSection]:
        # transcribe sections
    transcript_sections = []
    total_number_of_sections = len(audio_sections)
    for section_number, section in enumerate(audio_sections):
        print(f"Transscribing section {section_number} of {total_number_of_sections}.")
        transcription = whisper(
            section["audio"].numpy(),
            chunk_length_s=30,
            stride_length_s=5,
            batch_size=8,
        )
        print(transcription["text"])
        transcript_section = TranscriptSection(
                start=section["turn"].start,
                end=section["turn"].end,
                speaker=section["speaker"],
                text=transcription["text"].strip(),
            )
        transcript_sections.append(transcript_section)
    return transcript_sections


def save_transcript_as_txt(transcript: list[TranscriptSection], file_name: Path):
    with open(file_name, "w") as file:
        for transcript_section in transcript:
            file.write(
                f"start={transcript_section.start:.1f}s "
                + f"end={transcript_section.end:.1f}s "
                + f"speaker={transcript_section.speaker}:\n"
            )
            file.write(transcript_section.text)
            file.write("\n")
