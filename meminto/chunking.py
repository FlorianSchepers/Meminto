from meminto.llm.tokenizers import Tokenizer
from meminto.transcriber import TranscriptSection

RATIO_OF_TOKENS_RESERVED_FOR_RESPONSE = 0.3


def chunk_transcript_old(
    system_prompt: str,
    transcript: list[TranscriptSection],
    tokenizer: Tokenizer,
    max_tokens: int,
) -> list[str]:
    average_number_of_tokens_per_chunk = _get_average_number_of_tokens_per_chunk(
        system_prompt=system_prompt,
        user_prompt="".join(map(str, transcript)),
        tokenizer=tokenizer,
        max_tokens=max_tokens,
    )

    transcript_chunks = []
    current_chunk = ""
    for transcript_section in transcript:
        current_chunk += str(transcript_section)
        if (
            tokenizer.number_of_tokens(current_chunk)
            >= average_number_of_tokens_per_chunk
        ):
            transcript_chunks.append(current_chunk)
            current_chunk = ""
    if current_chunk:
        transcript_chunks.append(current_chunk)
    return transcript_chunks


def chunk_transcript(
    system_prompt: str,
    transcript: list[TranscriptSection],
    tokenizer: Tokenizer,
    max_tokens: int,
) -> list[str]:
    upper_limit = _get_average_number_of_tokens_per_chunk(
        system_prompt=system_prompt,
        user_prompt="".join(map(str, transcript)),
        tokenizer=tokenizer,
        max_tokens=max_tokens,
    )
    transcript_as_strings = [str(section) for section in transcript]
    lower_limit = 0.4 * upper_limit

    transcript_chunks: list[list[str]] = []
    current_chunk: list[str] = []
    for section in transcript_as_strings:
        current_chunk_size = tokenizer.number_of_tokens("".join(current_chunk))
        section_size = tokenizer.number_of_tokens(section)
        if current_chunk_size + section_size > upper_limit:
            if current_chunk:
                transcript_chunks.append(current_chunk)
            current_chunk = [section]
        else:
            current_chunk.append(section)

    current_chunk_size = tokenizer.number_of_tokens("".join(current_chunk))
    if current_chunk_size > lower_limit:
        transcript_chunks.append(current_chunk)
    else:
        while current_chunk_size < lower_limit:
            previous_chunk = transcript_chunks.pop()
            current_chunk.insert(0, previous_chunk.pop())
            transcript_chunks.append(previous_chunk)
            current_chunk_size = tokenizer.number_of_tokens("".join(current_chunk))
        transcript_chunks.append(current_chunk)

    return ["".join(chunk) for chunk in transcript_chunks]


def _get_average_number_of_tokens_per_chunk(
    system_prompt: str,
    user_prompt: str,
    tokenizer: Tokenizer,
    max_tokens: int,
) -> int:
    token_count_system_prompt = tokenizer.number_of_tokens(system_prompt)
    token_count_available_for_inference = int(max_tokens) - token_count_system_prompt
    token_count_reserved_for_response = int(
        token_count_available_for_inference * RATIO_OF_TOKENS_RESERVED_FOR_RESPONSE
    )
    token_count_available_for_user_prompt = (
        token_count_available_for_inference - token_count_reserved_for_response
    )

    token_count_user_prompt = tokenizer.number_of_tokens(user_prompt)
    number_of_chunks = (
        token_count_user_prompt // token_count_available_for_user_prompt + 1
    )
    average_number_of_tokens_per_chunk = token_count_user_prompt // number_of_chunks + 1

    print("Spliting user prompt in chunks:")
    print(f"LLM max. token count: {max_tokens}")
    print(f"Token count of system prompt: {token_count_system_prompt}")
    print(f"Token count reserved for response: {token_count_reserved_for_response}")
    print(
        f"Token count available for user prompt: {token_count_available_for_user_prompt}"
    )
    print(f"Token count of user prompt: {token_count_user_prompt}")
    print(f"Number of chunks: {number_of_chunks}")
    print(f"Average number of tokens per chunk: {average_number_of_tokens_per_chunk}")

    return average_number_of_tokens_per_chunk


def _get_token_count_available_for_user_prompt(
    system_prompt: str,
    user_prompt: str,
    tokenizer: Tokenizer,
    max_tokens: int,
) -> int:
    token_count_system_prompt = tokenizer.number_of_tokens(system_prompt)
    token_count_available_for_inference = int(max_tokens) - token_count_system_prompt
    token_count_reserved_for_response = int(
        token_count_available_for_inference * RATIO_OF_TOKENS_RESERVED_FOR_RESPONSE
    )
    token_count_available_for_user_prompt = (
        token_count_available_for_inference - token_count_reserved_for_response
    )

    return token_count_available_for_user_prompt
