import os
from typing import Tuple
from decorators import log_time
from helpers import Language
from meminto.llm.llm import LLM
from prompts import (
    CONTEXT,
    EXAMPLE_INPUT,
    EXAMPLE_INPUT_INTRO,
    EXAMPLE_OUTPUT,
    EXAMPLE_OUTPUT_INTRO,
    INSTRUCTIONS_CREATE_MEETING_MINUTES,
    INSTRUCTIONS_MERGE_MEETING_MINUTES,
    SELECT_LANGUAGE,
)
from llm.tokenizers import get_token_count
from meminto.transcriber import TranscriptSection

RATIO_OF_TOKENS_RESERVED_FOR_RESPONSE = 0.3


def get_number_of_tokens_per_chunk(
    system_prompt: str, transcript: list[TranscriptSection], model: str
) -> int:
    max_tokens = os.environ["LLM_MAX_TOKENS"]
    token_count_system_prompt = get_token_count(system_prompt, model)
    token_count_transcript = get_token_count("".join(map(str, transcript)), model)
    token_count_available = int(max_tokens) - token_count_system_prompt
    token_count_reserved_for_response = int(
        token_count_available * RATIO_OF_TOKENS_RESERVED_FOR_RESPONSE
    )
    token_count_reserved_for_response = int(
        token_count_available * RATIO_OF_TOKENS_RESERVED_FOR_RESPONSE
    )
    token_count_per_chunk = token_count_available - token_count_reserved_for_response

    number_of_chunks = token_count_transcript // token_count_per_chunk + 1
    number_of_tokens_per_chunk = token_count_transcript // number_of_chunks + 1

    print(f"Batching transcript:")
    print(f"LLM max. token count: {max_tokens}")
    print(f"Token count of system prompt: {token_count_system_prompt}")
    print(f"Token count reserved for response: {token_count_reserved_for_response}")
    print(f"Token count per transcript chunk: {token_count_per_chunk}")
    print(f"Token count of transcript: {token_count_transcript}")
    print(f"Number of chunks: {number_of_chunks}")
    print(f"Number of tokens per chunk: {number_of_tokens_per_chunk}")

    return number_of_tokens_per_chunk


def split_transcript_in_chunks(
    system_prompt: str, transcript: list[TranscriptSection], model: str
) -> list[list[TranscriptSection]]:
    number_of_tokens_per_chunk = get_number_of_tokens_per_chunk(
        system_prompt, transcript, model
    )

    transcript_chunks = []
    chunks = []
    for transcript_section in transcript:
        chunks.append(transcript_section)
        if (
            get_token_count("".join(map(str, chunks)), model)
            >= number_of_tokens_per_chunk
        ):
            transcript_chunks.append(chunks)
            chunks = []
    transcript_chunks.append(chunks)

    return transcript_chunks


def generate_meeting_minutes_chunks(
    transcript: list[TranscriptSection], language: Language, llm: LLM
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

    transcript_chunks = split_transcript_in_chunks(system_prompt, transcript, llm.model)

    meeting_minutes_chunks = []
    for chunk in transcript_chunks:
        text_chunk = "".join(map(str, chunk))
        meeting_minutes_chunk = llm.infer(system_prompt, text_chunk)
        meeting_minutes_chunks.append(meeting_minutes_chunk)

    return meeting_minutes_chunks


def meeting_minutes_chunks_to_text(meeting_minutes_chunks: list[str]) -> str:
    meeting_minutes_chunks_as_text = ""
    for idx, chunks in enumerate(meeting_minutes_chunks):
        meeting_minutes_chunks_as_text = (
            meeting_minutes_chunks_as_text + f"Section {idx+1}\n" + chunks + "\n\n"
        )
    return meeting_minutes_chunks_as_text


def get_merged_meeting_minutes(
    meeting_minutes_chunks: list[str], language: Language, llm: LLM
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
                meeting_minutes_chunks_as_text = meeting_minutes_chunks_to_text(
                    meeting_minutes_chunks[i : i + 2]
                )
                token_count_system_prompt = get_token_count(system_prompt, llm.model)
                token_count_meeting_minutes = get_token_count(
                    meeting_minutes_chunks_as_text, llm.model
                )
                print("Mergin Batches: ")
                print(f"Token count of system prompt: {token_count_system_prompt}")
                print(
                    f"Token count of meeting minutes chunks: {token_count_meeting_minutes}"
                )
                print(
                    f"Total token count: {token_count_system_prompt + token_count_meeting_minutes}"
                )
                merged_minutes = llm.infer(
                    system_prompt, meeting_minutes_chunks_as_text
                )
                merged_meeting_minutes.append(merged_minutes)
            elif i < len(meeting_minutes_chunks):
                merged_meeting_minutes.append(meeting_minutes_chunks[i])
        meeting_minutes_chunks = merged_meeting_minutes
    return meeting_minutes_chunks[0]


@log_time
def transcript_to_meeting_minutes(
    transcript: list[TranscriptSection], language: Language
) -> Tuple[str, list[str]]:
    llm = LLM(
        model=str(os.environ["LLM_MODEL"]),
        url=os.environ["LLM_URL"],
        authorization=os.environ["LLM_AUTHORIZATION"],
        temperature=0.5,
        max_tokens=os.environ["LLM_MAX_TOKENS"],
    )

    meeting_minutes_chunks = generate_meeting_minutes_chunks(transcript, language, llm)
    meeting_minutes_chunks_as_text = meeting_minutes_chunks_to_text(
        meeting_minutes_chunks
    )
    print(
        "----------------------------- meeting_minutes_chunks_as_text ------------------------------------"
    )
    print(meeting_minutes_chunks_as_text)
    merged_meeting_minutes = ""
    merged_meeting_minutes = get_merged_meeting_minutes(
        meeting_minutes_chunks, language, llm
    )
    print(
        "----------------------------- merged_meeting_minutes ------------------------------------"
    )
    print(merged_meeting_minutes)

    return (merged_meeting_minutes, meeting_minutes_chunks)
