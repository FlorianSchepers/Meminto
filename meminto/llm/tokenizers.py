import os
import tiktoken
from sentencepiece import SentencePieceProcessor
from transformers import AutoTokenizer, OpenAIGPTTokenizer


class Tokenizer:
    def __init__(self, model: str):
        self.model = model
        self.tokenizer = self._select_tokenizer()

    def _select_tokenizer(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model)
        except:
            if self.model in tiktoken.model.MODEL_TO_ENCODING.keys():
                tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    "meta-llama/Llama-2-70b-chat-hf"
                )  # Use Llama tokenizer as conservative fallback

        print(f'Using tokenizer for model: {tokenizer.name_or_path}')
        return tokenizer
    
    def tokenize(self, content: str)->list[str]:
        return self.tokenizer.tokenize(content)
    
    def number_of_tokens(self, content:str)->int:
        tokens = self.tokenize(content)
        return len(tokens)
