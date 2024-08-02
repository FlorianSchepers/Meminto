import tiktoken
from transformers import AutoTokenizer, OpenAIGPTTokenizer
from huggingface_hub import login


class Tokenizer:
    def __init__(self, model: str, hugging_face_acces_token: str):
        self.model = model
        self.hugging_face_acces_token = hugging_face_acces_token
        self.tokenizer = self._select_tokenizer()

    def _select_tokenizer(self):
        login(token=self.hugging_face_acces_token)
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model)
        except Exception as e:
            if self.model in tiktoken.model.MODEL_TO_ENCODING.keys():
                tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
            else:
                print(
                    f'\n'
                    f'Could not find a matching tokenizer for the model "{self.model}".\n'
                    f'Error message:\n' 
                    f'{e}\n'
                    f'\n'
                    f'Will try to use the tokenizer for "meta-llama/Llama-2-70b-chat-hf" as fallback instead.\n'
                ) 
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        "meta-llama/Llama-2-70b-chat-hf"
                    )
                except Exception as e:
                    tokenizer = None
                    print(
                        f'\n'
                        f'Failed to download "meta-llama/Llama-2-70b-chat-hf".\n'
                        f'Error message:\n' 
                        f'{e}\n'
                        f'\n'
                        f'Will assume 3 characters = 1 token as fallback.\n'
                        
                    )
        return tokenizer

    def tokenize(self, content: str) -> list[str]:
        return self.tokenizer.tokenize(content)

    def number_of_tokens(self, content: str) -> int:
        if self.tokenizer:
            tokens = self.tokenize(content)
            token_count = len(tokens)
        else:
            token_count = len(content)//3+1
        return token_count
