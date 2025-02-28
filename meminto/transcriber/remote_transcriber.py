import logging
from pathlib import Path
from typing import Any

import requests
from meminto.logging_config import log_time
from meminto.transcriber.base_transcriber import BaseTranscriber
from meminto.transcriber.model import Transcript, TranscriptSection


class RemoteTranscriber (BaseTranscriber):
    def __init__(
        self,
        url: str,
        authorization: str,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.url = url
        self.authorization = authorization

    @log_time  
    def create_transcript(self, audio_file: Path) -> Transcript:
        
        transcription_response = self.send_request(audio_file)

        transcript_sections = []
        for segment in transcription_response["segments"]:
            transcript_section = TranscriptSection(
                start=segment["start"],
                end=segment["end"],
                speaker=segment["speaker"],
                text=segment["text"],
            )
            transcript_sections.append(transcript_section)

        return Transcript(
            text=transcription_response["text"],
            language=transcription_response["language"],
            sections=transcript_sections,
        )
    
    def send_request(self, audio_file: Path) -> Any:
        headers = {
            "accept": 'application/json',
            "Authorization": f'Bearer {self.authorization}',
        }
        files = {
            "file": (audio_file.name, audio_file.open('rb'), "audio/vnd.wave"),
        }
        data = {
            'model': 'large-v3',
            'response_format': 'verbose_json',
            'timestamp_granularity': 'segment,word',
            'speaker_diarization': 'true',
        }

        self.logger.info(f"Endpoint used for transcription: {self.url}")
        
        response = requests.post(url=self.url, headers=headers, files=files, data=data)
        
        self.logger.debug(f"Endpoint response: {response}")
        self.logger.debug(f"Endpoint response text: {response.text}")
        self.logger.debug(f"Endpoint response json: {response.json()}")
        
        return response.json()

