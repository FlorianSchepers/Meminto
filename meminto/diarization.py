import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from decorators import log_time

@log_time
def diarize_audio(audio_source: Path) -> Any:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1", use_auth_token=os.environ["HUGGING_FACE_ACCESS_TOKEN"]
    )
    return pipeline(audio_source)