from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
import io
import json
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
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
from torch import Tensor
import soundfile as sf


class WHISPER_MODEL_SIZE(Enum):
    tiny = "tiny"
    base = "base"
    small = "small"
    medium = "medium"
    distil = "distil"
    large = "large"


@dataclass
class TranscriptChunk:
    timestamp: Tuple[float, float]
    text: str
    speaker: Optional[str] = None

    def __str__(self):
        speaker_str = self.speaker if self.speaker is not None else "Unknown Speaker"
        return f"[{self.timestamp[0]:.2f} - {self.timestamp[1]:.2f}] {speaker_str}: {self.text}"


@dataclass
class Transcript:
    text: str
    chunks: List[TranscriptChunk] = field(default_factory=list)

    def __str__(self):
        return "\n".join(str(chunk) for chunk in self.chunks)


class Transcriber:
    @log_time
    def transcribe(self, audio: Union[Tensor, np.ndarray]):
        raise NotImplementedError("Subclasses should implement this method")


class LocalTranscriber(Transcriber):
    def __init__(
        self,
        model_size: WHISPER_MODEL_SIZE = WHISPER_MODEL_SIZE.medium,
        english_only: bool = False,
    ):
        model_name = f"openai/whisper-{model_size.value}{'-en' if english_only else ''}"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Running on {device}")

        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
            chunk_length_s=30,
            stride_length_s=5,
            return_timestamps=True,
        )

    def transcribe(self, audio: Union[Tensor, np.ndarray]) -> Transcript:
        """
        Transcribe the audio tensor using the ASR pipeline.

        :param audio: Audio data as a Tensor or numpy array of shape (samples,)
        :return: Transcription result with text and timestamps
        """
        audio_input = audio.numpy() if isinstance(audio, Tensor) else audio
        result = self.asr_pipeline(
            audio_input, chunk_length_s=30, return_timestamps=True
        )

        chunks_data = result.get("chunks", [])
        chunks = [
            TranscriptChunk(
                timestamp=chunk["timestamp"],
                text=chunk["text"],
            )
            for chunk in chunks_data
        ]

        return Transcript(
            text=result.get("text", ""),
            chunks=chunks,
        )
    
    
class RemoteTranscriber(Transcriber):
    def __init__(
        self,
        url: str,
        authorization: str,
    ):
        self.url = url
        self.authorization = authorization


    @log_time
    def transcribe(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        sample_rate: int = 16000
    ) -> dict:
        headers = {
            "Authorization": self.authorization,
        }

        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        if audio.ndim > 1:
            audio = np.squeeze(audio)

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, audio, sample_rate, format='WAV')
            wav_buffer.seek(0)

            files = {
                "file": ("audio.wav", wav_buffer, "audio/wav"),
                "response_format": (None, "verbose_json")
            }

            print(f"Endpoint used for transcription: {self.url}")
            response = requests.post(url=self.url, headers=headers, files=files)

        # Check for HTTP errors
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            print(f"Response content: {response.text}")
            return {}
        
        transcription_response = response.json()

        segmentes = transcription_response["segments"]

        chunks = []
        for segment in segmentes:
            chunk = TranscriptChunk(
                timestamp = (segment["start"], segment["end"]),
                text = segment["text"].strip()
            )
            chunks.append(chunk)

        return Transcript(
            text = transcription_response["text"].strip(),
            chunks=chunks,
        )