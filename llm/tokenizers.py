import tiktoken
from sentencepiece import SentencePieceProcessor

from llm.llm_inference import LLM_MODEL


def get_tiktoken_count(content, model):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(content))


def get_sentencepiece_count(content):
    sp_model = SentencePieceProcessor("llm/tokenizer.model")
    return len(sp_model.encode(content))


def get_token_count(content, model):
    if model in tiktoken.model.MODEL_TO_ENCODING.keys():
        return get_tiktoken_count(content, model)
    elif model in {"llama", "llama2", "Llama-2-70b-chat-hf"}:
        return get_sentencepiece_count(content)
    else:
        return get_sentencepiece_count(content)


def get_token_count_for_inference(system_prompt, user_prompt):
    token_count_system_prompt = get_token_count(system_prompt, LLM_MODEL)
    token_count_transcript = get_token_count("".join(map(str, user_prompt)), LLM_MODEL)

    print("Token count for LLM inference: ")
    print(f"Token count of system prompt: {token_count_system_prompt}")
    print(f"Token count of transcript: {token_count_transcript}")
    print(f"Total token count: {token_count_system_prompt + token_count_transcript}")

    return (token_count_system_prompt, token_count_transcript)
