from dataclasses import dataclass
from pathlib import Path
import torchaudio as torchaudio
from torch import Tensor
from meminto.decorators import log_time
from pyannote.core import Annotation, Segment
from pyannote.core.utils.types import Label


SAMPLING_RATE = 16000


@dataclass
class AudioSection:
    speaker: Label
    turn: Segment
    audio: Tensor


def load_audio(audio_path: Path) -> Tensor:
    audio, sr = torchaudio.load(audio_path)
    number_of_channels = audio.size()[0]
    if number_of_channels > 1:
        audio = audio[0]
    resampler = torchaudio.transforms.Resample(sr, SAMPLING_RATE)
    audio_resampled: Tensor = resampler(audio)
    return audio_resampled.squeeze()


@log_time
def split_audio(
    audio_input_file_path: Path, diarization: Annotation
) -> list[AudioSection]:
    audio = load_audio(audio_input_file_path)

    audio_sections = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        audio_sections.append(
            AudioSection(
                speaker=speaker,
                turn=turn,
                audio=audio[
                    int(SAMPLING_RATE * round(turn.start, 1)) : int(
                        SAMPLING_RATE * round(turn.end, 1)
                    )
                ],
            )
        )
    return audio_sections

@log_time
def split_audio_dict(
    audio_input_file_path: Path, diarization_list: list[dict]
) -> list[AudioSection]:
    audio = load_audio(audio_input_file_path)

    audio_sections = []
    for segment in diarization_list:
        audio_sections.append(
            AudioSection(
                speaker=segment["speaker"],
                turn=segment["turn"],
                audio=audio[
                    int(SAMPLING_RATE * round(segment["start"], 1)) : int(
                        SAMPLING_RATE * round(segment["end"], 1)
                    )
                ],
            )
        )
    return audio_sections
