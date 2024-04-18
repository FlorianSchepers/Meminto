import tiktoken
from sentencepiece import SentencePieceProcessor

LLAMA_MODELS = {"llama", "llama2", "Llama-2-70b-chat-hf", "Mixtral-8x7B-Instruct-v0.1"}

def get_tiktoken_count(content: str, model: str) -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(content))


def get_sentencepiece_count(content: str) -> int:
    sp_model = SentencePieceProcessor("meminto/llm/tokenizer.model")
    return len(sp_model.encode(content))


def get_token_count(content: str, model: str) -> int:
    if model in tiktoken.model.MODEL_TO_ENCODING.keys():
        return get_tiktoken_count(content, model)
    elif model in LLAMA_MODELS:
        return get_sentencepiece_count(content)
    else:
        # If model is unknown use sentencepiece as it is the more conservative estimate 
        return get_sentencepiece_count(content) 
