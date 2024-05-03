from meminto.llm.tokenizers import Tokenizer
from meminto.transcriber import TranscriptSection

RATIO_OF_TOKENS_RESERVED_FOR_RESPONSE = 0.3

def chunk_transcript(
          system_prompt: str, transcript: list[TranscriptSection], tokenizer: Tokenizer, max_tokens: int
    ) -> list[str]:
        number_of_tokens_per_chunk = _number_of_tokens_per_chunk(
            system_prompt=system_prompt, transcript=transcript, tokenizer=tokenizer, max_tokens=max_tokens
        )

        transcript_chunks = []
        current_chunk = ""
        for transcript_section in transcript:
            current_chunk = current_chunk + str(transcript_section)
            if (
                tokenizer.number_of_tokens(current_chunk + str(transcript_section))
                >= number_of_tokens_per_chunk
            ):
                transcript_chunks.append(current_chunk)
                current_chunk = ""
        if current_chunk: 
            transcript_chunks.append(current_chunk) 
        return transcript_chunks 

def _number_of_tokens_per_chunk(
        system_prompt: str, transcript: list[TranscriptSection], tokenizer: Tokenizer, max_tokens:int
    ) -> int:
        token_count_system_prompt = tokenizer.number_of_tokens(system_prompt)
        token_count_available = int(max_tokens) - token_count_system_prompt
        token_count_reserved_for_response = int(
            token_count_available * RATIO_OF_TOKENS_RESERVED_FOR_RESPONSE
        )
        token_count_per_chunk = token_count_available - token_count_reserved_for_response

        token_count_transcript = tokenizer.number_of_tokens("".join(map(str, transcript)))
        number_of_chunks = token_count_transcript // token_count_per_chunk + 1
        number_of_tokens_per_chunk = token_count_transcript // number_of_chunks + 1

        print("Spliting transcript in chunks:")
        print(f"LLM max. token count: {max_tokens}")
        print(f"Token count of system prompt: {token_count_system_prompt}")
        print(f"Token count reserved for response: {token_count_reserved_for_response}")
        print(f"Token count per transcript chunk: {token_count_per_chunk}")
        print(f"Token count of transcript: {token_count_transcript}")
        print(f"Number of chunks: {number_of_chunks}")
        print(f"Number of tokens per chunk: {number_of_tokens_per_chunk}")

        return number_of_tokens_per_chunk