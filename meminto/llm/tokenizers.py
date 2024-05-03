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
        except(Exception):
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
