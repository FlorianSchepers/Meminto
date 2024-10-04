from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import torchaudio as torchaudio
from torch import Tensor

from meminto.transcriber_new import TranscriptChunk

def load_audio(audio_path: Union[str, Path], target_sampling_rate: int = 16000) -> Tensor:
    audio, sr = torchaudio.load(str(audio_path))
    print(f"Original Sampling Rate: {sr}")
    
    # Convert to mono by averaging channels if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    # Resample if the original sampling rate is different from the target
    if sr != target_sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sampling_rate)
        audio = resampler(audio)
    
    return audio.squeeze()

@dataclass
class AudioSegment:
    timestamp: Tuple[float, float]
    audio: np.ndarray

def extract_audio_segments(
    audio: Union[Tensor, np.ndarray],
    sampling_rate: int,
    timestamps: List[Tuple[float, float]],
) -> List[AudioSegment]:
    """
    Extract audio segments based on the timestamps from the transcription.
    
    :param audio: Audio data as a Tensor or numpy array of shape (samples,)
    :param sr: Sampling rate of the audio
    :param segments: List of transcription segments with timestamps
    :return: List of audio segments with associated metadata
    """
    if isinstance(audio, Tensor):
        audio_data = audio.numpy()
    else:
        audio_data = audio
    audio_segments: List[AudioSegment] = []
    
    for timestamp in timestamps:
        start_sample: int = int(timestamp[0] * sampling_rate)
        end_sample: int = int(timestamp[1] * sampling_rate)
        audio_segments.append(
            AudioSegment(
                timestamp=timestamp, 
                audio=audio_data[start_sample:end_sample]
                )
            )
    return audio_segments