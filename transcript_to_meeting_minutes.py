import os
from decorators import log_time
from llm import get_chat_completion
from prompts import (
    AI_SUGGESTIONS,
    CONTEXT,
    EXAMPLE_1,
    EXAMPLE_1_AI_SUGGESTIONS,
    EXAMPLE_INPUT,
    EXAMPLE_INPUT_INTRO,
    EXAMPLE_INTRO,
    INSTRUCTIONS_CREATE_MEETING_MINUTES,
    INSTRUCTIONS_MERGE_MEETING_MINUTES,
    SELECT_LANGUAGE,
)
from tokenizer_helpers import get_token_count
from transcription import TranscriptSection


def get_batched_transcript(system_prompt, transcript: list[TranscriptSection]):
    token_count_system_prompt = get_token_count(system_prompt)
    token_count_transcript = get_token_count("".join(map(str, transcript)))
    llm_max_tokens = (
        int(os.environ["LLM_MAX_TOKENS"]) if "LLM_MAX_TOKENS" in os.environ else 4000
    )
    token_count_available = int((llm_max_tokens - token_count_system_prompt) * 0.5)
    number_of_parts = token_count_transcript // token_count_available + 1
    token_per_part = token_count_transcript // number_of_parts + 1

    print(f"token_count_system_prompt: {token_count_system_prompt}")
    print(f"token_count_transcript: {token_count_transcript}")
    print(f"llm_max_tokens: {llm_max_tokens}")
    print(f"token_count_available: {token_count_available}")
    print(f"number_of_parts: {number_of_parts}")
    print(f"token_per_part: {token_per_part}")

    batched_transcript = []
    batch = []
    for transcript_section in transcript:
        batch.append(transcript_section)
        if get_token_count("".join(map(str, batch))) >= token_per_part:
            batched_transcript.append(batch)
            batch = []
    batched_transcript.append(batch)

    return batched_transcript


@log_time
def transcript_to_meeting_minutes(
    transcript: list[TranscriptSection], language
):
    system_prompt = (
        CONTEXT
        + INSTRUCTIONS_CREATE_MEETING_MINUTES
        + AI_SUGGESTIONS
        + SELECT_LANGUAGE
        + language
        + ".\n"
        + EXAMPLE_INTRO
        + EXAMPLE_1
        + EXAMPLE_1_AI_SUGGESTIONS
    )

    batched_transcript = get_batched_transcript(system_prompt, transcript)

    batched_meeting_minutes = []
    for batch in batched_transcript:
        batch_txt = ''.join(map(str, batch))
        batched_meeting_minutes.append(
            get_chat_completion(system_prompt, batch_txt)
        )

    return batched_meeting_minutes

def get_merged_meeting_minutes(batched_meeting_minutes, language):
    system_prompt = (
        CONTEXT
        + INSTRUCTIONS_MERGE_MEETING_MINUTES
        + AI_SUGGESTIONS
        + SELECT_LANGUAGE
        + language
        + ".\n"
        + EXAMPLE_INPUT_INTRO
        + EXAMPLE_INPUT
        + EXAMPLE_INTRO
        + EXAMPLE_1
        + EXAMPLE_1_AI_SUGGESTIONS
    )
    batched_meeting_minutes_txt = ""
    for idx, batch in enumerate(batched_meeting_minutes):
        batched_meeting_minutes_txt = batched_meeting_minutes_txt + f"Part {idx+1}\n" + batch + "\n\n"
    print(batched_meeting_minutes_txt)
    print()

    if len(batched_meeting_minutes) == 1:
        return batched_meeting_minutes[0]
    
    token_count_system_prompt = get_token_count(system_prompt)
    token_count_transcript = get_token_count("".join(map(str, batched_meeting_minutes_txt)))
    print(f"token_count_system_prompt: {token_count_system_prompt}")
    print(f"token_count_transcript: {token_count_transcript}")
    return get_chat_completion(system_prompt, batched_meeting_minutes_txt)