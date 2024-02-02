from typing import Tuple
from decorators import log_time
from helpers import Language
from llm.llm_inference import LLM_MAX_TOKENS, LLM_MODEL, infer_llm
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
from transcription import TranscriptSection

RATIO_OF_TOKENS_RESERVED_FOR_RESPONSE = 0.3


def get_number_of_tokens_per_batch(
    system_prompt: str, transcript: list[TranscriptSection]
) -> int:
    token_count_system_prompt = get_token_count(system_prompt, LLM_MODEL)
    token_count_transcript = get_token_count("".join(map(str, transcript)), LLM_MODEL)
    token_count_available = (
        int(LLM_MAX_TOKENS)
        - token_count_system_prompt
    )
    token_count_reserved_for_response = int(token_count_available*RATIO_OF_TOKENS_RESERVED_FOR_RESPONSE)
    token_count_per_batch = token_count_available - token_count_reserved_for_response
    
    number_of_batches = token_count_transcript // token_count_per_batch + 1
    number_of_tokens_per_batch = token_count_transcript // number_of_batches + 1

    print(f"Batching transcript:")
    print(f"LLM max. token count: {LLM_MAX_TOKENS}")
    print(f"Token count of system prompt: {token_count_system_prompt}")
    print(f"Token count reserved for response: {token_count_reserved_for_response}")
    print(f"Token count per transcript batch: {token_count_per_batch}")
    print(f"Token count of transcript: {token_count_transcript}")
    print(f"Number of batches: {number_of_batches}")
    print(f"Number of tokens per batch: {number_of_tokens_per_batch}")

    return number_of_tokens_per_batch


def get_batched_transcript(
    system_prompt: str, transcript: list[TranscriptSection]
) -> list[list[TranscriptSection]]:
    number_of_tokens_per_batch = get_number_of_tokens_per_batch(
        system_prompt, transcript
    )

    batched_transcript = []
    batch = []
    for transcript_section in transcript:
        batch.append(transcript_section)
        if (
            get_token_count("".join(map(str, batch)), LLM_MODEL)
            >= number_of_tokens_per_batch
        ):
            batched_transcript.append(batch)
            batch = []
    batched_transcript.append(batch)

    return batched_transcript


def get_batched_meeting_minutes(
    transcript: list[TranscriptSection], language: Language
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

    batched_transcript = get_batched_transcript(system_prompt, transcript)

    batched_meeting_minutes = []
    for batch in batched_transcript:
        batch_txt = "".join(map(str, batch))
        batched_meeting_minutes.append(infer_llm(system_prompt, batch_txt))

    return batched_meeting_minutes


def batched_meeting_minutes_to_text(batched_meeting_minutes: list[str]) -> str:
    batched_meeting_minutes_txt = ""
    for idx, batch in enumerate(batched_meeting_minutes):
        batched_meeting_minutes_txt = (
            batched_meeting_minutes_txt + f"Section {idx+1}\n" + batch + "\n\n"
        )
    return batched_meeting_minutes_txt


def get_merged_meeting_minutes(
    batched_meeting_minutes: list[str], language: Language
) -> str:
    if len(batched_meeting_minutes) == 1:
        return batched_meeting_minutes[0]

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

    batched_meeting_minutes_as_text = batched_meeting_minutes_to_text(
        batched_meeting_minutes
    )

    token_count_system_prompt = get_token_count(system_prompt, LLM_MODEL)
    token_count_meeting_minutes = get_token_count(batched_meeting_minutes_as_text, LLM_MODEL)

    print("Mergin Batches: ")
    print(f"Token count of system prompt: {token_count_system_prompt}")
    print(f"Token count of batched meeting minutes: {token_count_meeting_minutes}")
    print(f"Total token count: {token_count_system_prompt + token_count_meeting_minutes}")

    return infer_llm(system_prompt, batched_meeting_minutes_as_text)


@log_time
def transcript_to_meeting_minutes(
    transcript: list[TranscriptSection], language: Language
) -> Tuple[str, list[str]]:
    batched_meeting_minutes = get_batched_meeting_minutes(transcript, language)
    batched_meeting_minutes_as_text = batched_meeting_minutes_to_text(
        batched_meeting_minutes
    )
    print(
        "----------------------------- batched_meeting_minutes_as_text ------------------------------------"
    )
    print(batched_meeting_minutes_as_text)
    merged_meeting_minutes = ""
    merged_meeting_minutes = get_merged_meeting_minutes(
        batched_meeting_minutes, language
    )
    print(
        "----------------------------- merged_meeting_minutes ------------------------------------"
    )
    print(merged_meeting_minutes)

    return (merged_meeting_minutes, batched_meeting_minutes)
