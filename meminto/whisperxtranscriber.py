from typing import List
import torch
from meminto.decorators import log_time
from meminto.audio_processing import AudioSection
import whisperx

from meminto.transcriber import Transcriber, TranscriptSection


class WhisperXTranscriber(Transcriber):
    def __init__(
        self,
        compute_type: str = "float32",
        device: str = "cpu",
    ):
        if device == "gpu" and not torch.cuda.is_available():
            raise ValueError("GPU is not available")
        if device == "cpu" and compute_type == "float16":
            raise ValueError("Cannot use float16 on CPU")

        self.device = device
        self.compute_type = compute_type

    @log_time
    def transcribe(self, audio_sections: List[AudioSection]) -> List[TranscriptSection]:
        # Load the Whisper model with WhisperX
        model = whisperx.load_model(
            "large-v2", device=self.device, compute_type=self.compute_type
        )

        # Load the diarization model (optional if speaker diarization is not needed)
        diarize_model = whisperx.DiarizationPipeline(
            device=self.device, use_auth_token=None
        )
        audio_path = "/home/FlorianTNG/TNG/Fortbildung/Eigenes/Meeting_Minutes_AI/Meminto/examples/Scoreboard.wav"
        audio = whisperx.load_audio(audio_path)

        # Transcribe the audio
        result = model.transcribe(audio, batch_size=16)

        # Load alignment model and metadata
        language_code = result["language"]  # Extract detected language
        model_a, metadata = whisperx.load_align_model(
            language_code=language_code, device=self.device
        )

        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        # Perform speaker diarization
        diarize_segments = diarize_model(audio_path, min_speakers=2, max_speakers=2)

        # Map the segments to the speakers
        result = whisperx.assign_word_speakers(diarize_segments, result)
        transcript_sections = []
        for segment in result["segments"]:
            transcript_section = TranscriptSection(
                start=segment["start"],
                end=segment["end"],
                speaker=segment["speaker"],
                text=segment["text"],
            )
            transcript_sections.append(transcript_section)

        return transcript_sections
