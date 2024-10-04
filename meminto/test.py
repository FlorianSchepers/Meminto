#https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/speaker_verification.ipynb

import os
from pathlib import Path
from typing import Any, Dict, List, Union

from dotenv import load_dotenv

from meminto.audio_processing_new import extract_audio_segments, load_audio
from meminto.speaker_diarizer_new import SpeakerDiarizer
from meminto.transcriber_new import LocalTranscriber, RemoteTranscriber, Transcript

def update_transcription(transcript: Transcript, diarization: List[str]) -> Transcript:
    for idx, speaker in enumerate(diarization):
        transcript.chunks[idx].speaker = speaker
    return transcript

def main() -> None:
    load_dotenv()
    audio_path: Union[str, Path] = Path('examples\multivoice.wav')
    sampling_rate = 16000
    audio_tensor = load_audio(audio_path, sampling_rate)
    
    transcriber = RemoteTranscriber(
        url=os.environ["TRANSCRIBER_URL"],
        authorization=os.environ["TRANSCRIBER_AUTHORIZATION"],
    )
    # transcriber = LocalTranscriber()
    
    transcript = transcriber.transcribe(audio_tensor)
    
    audio_segments = extract_audio_segments(
        audio=audio_tensor, 
        sampling_rate=sampling_rate, 
        timestamps=[chunk.timestamp for chunk in transcript.chunks]
        )

    diarizer = SpeakerDiarizer(hugging_face_auth_token=os.environ["HUGGING_FACE_ACCESS_TOKEN"])
    speaker_lables = diarizer.diarize([audio_segment.audio for audio_segment in audio_segments], num_speakers=None)

    update_transcription(transcript, speaker_lables)

    print(transcript)

if __name__ == "__main__":
    main()