from enum import Enum


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


class WHISPER_X_MODELS(Enum):
    LARGE_V2 = "large-v2"


class PYANNOTE_AUDIO_MODELS(Enum):
    SPEAKER_DIARIZATION_2_1 = "pyannote/speaker-diarization@2.1"