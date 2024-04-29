import os
from typing import Tuple
from meminto.decorators import log_time
from meminto.helpers import Language
from meminto.llm.llm import LLM
from meminto.chunking import chunk_transcript
from meminto.prompts import (
    CONTEXT,
    EXAMPLE_INPUT,
    EXAMPLE_INPUT_INTRO,
    EXAMPLE_OUTPUT,
    EXAMPLE_OUTPUT_INTRO,
    INSTRUCTIONS_CREATE_MEETING_MINUTES,
    INSTRUCTIONS_MERGE_MEETING_MINUTES,
    SELECT_LANGUAGE,
)
from meminto.llm.tokenizers import Tokenizer
from meminto.transcriber import TranscriptSection
from huggingface_hub import login


class MeetingMinutesGenerator:
    def __init__(
        self,
        tokenizer: Tokenizer,
        llm: LLM,
    ):
        self.tokenizer = tokenizer
        self.llm = llm

    @log_time
    def generate(self, transcript: list[TranscriptSection], language: Language) -> str:
        meeting_minutes_chunks = self.generate_meeting_minutes_chunks(
            transcript, language
        )

        merged_meeting_minutes = self.merged_meeting_minutes(
            meeting_minutes_chunks, language
        )

        return merged_meeting_minutes

    def generate_meeting_minutes_chunks(
        self,
        transcript: list[TranscriptSection],
        language: Language,
    ) -> list[str]:

        system_prompt = (
            CONTEXT
            + INSTRUCTIONS_CREATE_MEETING_MINUTES
            + SELECT_LANGUAGE
            + language.value
            + ".\n"
            + EXAMPLE_OUTPUT_INTRO
            + EXAMPLE_OUTPUT
        )

        transcript_chunks = chunk_transcript(
            system_prompt, transcript, self.tokenizer, self.llm.max_tokens
        )

        meeting_minutes_chunks = []
        for chunk in transcript_chunks:
            meeting_minutes_chunk = self.llm.infer(system_prompt, chunk)
            meeting_minutes_chunks.append(meeting_minutes_chunk)

        return meeting_minutes_chunks

    def merged_meeting_minutes(
        self,
        meeting_minutes_chunks: list[str],
        language: Language,
    ) -> str:
        system_prompt = (
            CONTEXT
            + INSTRUCTIONS_MERGE_MEETING_MINUTES
            + SELECT_LANGUAGE
            + language.value
            + ".\n"
            + EXAMPLE_INPUT_INTRO
            + EXAMPLE_INPUT
            + EXAMPLE_OUTPUT_INTRO
            + EXAMPLE_OUTPUT
        )

        while len(meeting_minutes_chunks) > 1:
            merged_meeting_minutes = []
            for i in range(0, len(meeting_minutes_chunks), 2):
                if i + 1 < len(meeting_minutes_chunks):
                    meeting_minutes_chunks_as_text = (
                        self.meeting_minutes_chunks_to_text(
                            meeting_minutes_chunks[i : i + 2]
                        )
                    )
                    token_count_system_prompt = self.tokenizer.number_of_tokens(
                        system_prompt
                    )
                    token_count_meeting_minutes = self.tokenizer.number_of_tokens(
                        meeting_minutes_chunks_as_text
                    )
                    print("Mergin meeting minutes chunks: ")
                    print(f"Token count of system prompt: {token_count_system_prompt}")
                    print(
                        f"Token count of meeting minutes chunks: {token_count_meeting_minutes}"
                    )
                    print(
                        f"Total token count: {token_count_system_prompt + token_count_meeting_minutes}"
                    )
                    merged_minutes = self.llm.infer(
                        system_prompt, meeting_minutes_chunks_as_text
                    )
                    merged_meeting_minutes.append(merged_minutes)
                elif i < len(meeting_minutes_chunks):
                    merged_meeting_minutes.append(meeting_minutes_chunks[i])
            meeting_minutes_chunks = merged_meeting_minutes
        return meeting_minutes_chunks[0]

    @staticmethod
    def meeting_minutes_chunks_to_text(meeting_minutes_chunks: list[str]) -> str:
        meeting_minutes_chunks_as_text = ""
        for idx, chunks in enumerate(meeting_minutes_chunks):
            meeting_minutes_chunks_as_text = (
                meeting_minutes_chunks_as_text + f"Section {idx+1}\n" + chunks + "\n\n"
            )
        return meeting_minutes_chunks_as_text
