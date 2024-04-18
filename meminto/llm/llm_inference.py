from dataclasses import asdict, dataclass
import os
from typing import TypedDict
import requests

from llm.tokenizers import LLAMA_MODELS

@dataclass
class Message:
    role: str
    content: str

@dataclass
class LLMParameters:
    model: str
    temperature: float
    max_tokens: int
    messages: list[Message]

Headers = TypedDict("Headers", {"Content-Type": str, "Authorization": str})

def get_llm_parameters(system_prompt: str, user_prompt: str) -> LLMParameters:
    llm_model = os.environ["LLM_MODEL"]
    if llm_model in LLAMA_MODELS:
        system_prompt = "<s>[INST] <<SYS>>\n" + system_prompt + "\n<</SYS>>\n\n"
        user_prompt = user_prompt + "[/INST]"

    messages: list[Message] = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ]

    max_tokens = (
        os.environ["LLM_MAX_TOKENS"] if "LLM_MAX_TOKENS" in os.environ else "4000"
    )   
    llm_parameters = LLMParameters(
        model=llm_model,
        temperature=0.5,
        max_tokens=int(max_tokens),
        messages=messages,
    )

    return llm_parameters


def get_headers() -> Headers:
    headers = {}
    headers["Content-Type"] = "application/json"
    headers["Authorization"] = os.environ["LLM_AUTHORIZATION"]
    return headers


def infer_llm(system_prompt: str, user_prompt: str) -> str:
    llm_parameters = get_llm_parameters(system_prompt, user_prompt)
    headers = get_headers()
    
    llm_url = os.environ["LLM_URL"]
    print(f"Url used for LLM request: {llm_url}")
    response = requests.post(url=llm_url, json=asdict(llm_parameters), headers=headers)
    return response.json()["choices"][0]["message"]["content"]
