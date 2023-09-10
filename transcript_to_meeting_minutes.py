from decorators import log_time
from llm_inference.llm_inference import LLM_MAX_TOKENS, LLM_MODEL, infer_llm
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
from llm_inference.tokenizers import get_token_count, get_token_count_for_inference
from transcription import TranscriptSection

TOKEN_COUNT_RESERVED_FOR_RESPONSE = 1000


def get_number_of_tokens_per_batch(system_prompt, transcript: list[TranscriptSection]):
    token_count_system_prompt = get_token_count(system_prompt, LLM_MODEL)
    token_count_transcript = get_token_count("".join(map(str, transcript)), LLM_MODEL)
    token_count_available = (
        int(LLM_MAX_TOKENS)
        - token_count_system_prompt
        - TOKEN_COUNT_RESERVED_FOR_RESPONSE
    )
    number_of_batches = token_count_transcript // token_count_available + 1
    number_of_tokens_per_batch = token_count_transcript // number_of_batches + 1

    print(f"Batching transcript:")
    print(f"LLM max. token count: {LLM_MAX_TOKENS}")
    print(f"Token count of system prompt: {token_count_system_prompt}")
    print(f"Token count reserved for response: {TOKEN_COUNT_RESERVED_FOR_RESPONSE}")
    print(f"Token count available for transcript: {token_count_available}")
    print(f"Token count of transcript: {token_count_transcript}")
    print(f"Number of batches: {number_of_batches}")
    print(f"Number of tokens per batch: {number_of_tokens_per_batch}")

    return number_of_tokens_per_batch


def get_batched_transcript(system_prompt, transcript: list[TranscriptSection]):
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


def get_batched_meeting_minutes(system_prompt, batched_transcript):
    batched_meeting_minutes = []
    for batch in batched_transcript:
        batch_txt = "".join(map(str, batch))
        batched_meeting_minutes.append(infer_llm(system_prompt, batch_txt))
    return batched_meeting_minutes


def batched_meeting_minutes_to_text(batched_meeting_minutes):
    batched_meeting_minutes_txt = ""
    for idx, batch in enumerate(batched_meeting_minutes):
        batched_meeting_minutes_txt = (
            batched_meeting_minutes_txt + f"Part {idx+1}\n" + batch + "\n\n"
        )
    return batched_meeting_minutes_txt


def get_merged_meeting_minutes(batched_meeting_minutes, language):
    if len(batched_meeting_minutes) == 1:
        return batched_meeting_minutes[0]

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

    batched_meeting_minutes_as_text = batched_meeting_minutes_to_text(
        batched_meeting_minutes
    )
    print(batched_meeting_minutes_as_text)
    print()

    get_token_count_for_inference(system_prompt, batched_meeting_minutes_as_text)
    return infer_llm(system_prompt, batched_meeting_minutes_as_text)


@log_time
def transcript_to_meeting_minutes(transcript: list[TranscriptSection], language):
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

    batched_meeting_minutes = get_batched_meeting_minutes(
        system_prompt, batched_transcript
    )

    merged_meeting_minutes = get_merged_meeting_minutes(
        batched_meeting_minutes, language
    )

    return (merged_meeting_minutes, batched_meeting_minutes)
