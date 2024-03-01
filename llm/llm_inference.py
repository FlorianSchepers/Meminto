from dataclasses import asdict, dataclass
import os
from typing import TypedDict
import requests

from llm.tokenizers import LLAMA_MODELS

LLM_URL = os.environ["LLM_URL"]
LLM_MODEL = os.environ["LLM_MODEL"]
LLM_AUTHORIZATION = os.environ["LLM_AUTHORIZATION"]
LLM_MAX_TOKENS = (
    os.environ["LLM_MAX_TOKENS"] if "LLM_MAX_TOKENS" in os.environ else "4000"
)

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
    if LLM_MODEL in LLAMA_MODELS:
        system_prompt = "<s>[INST] <<SYS>>\n" + system_prompt + "\n<</SYS>>\n\n"
        user_prompt = user_prompt + "[/INST]"

    messages: list[Message] = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ]

    llm_parameters = LLMParameters(
        model=LLM_MODEL,
        temperature=0,
        max_tokens=int(LLM_MAX_TOKENS),
        messages=messages,
    )

    return llm_parameters


def get_headers() -> Headers:
    headers = {}
    headers["Content-Type"] = "application/json"
    headers["Authorization"] = LLM_AUTHORIZATION
    return headers


def infer_llm(system_prompt: str, user_prompt: str) -> str:
    llm_parameters = get_llm_parameters(system_prompt, user_prompt)
    headers = get_headers()

    print(f"Url used for LLM request: {LLM_URL}")
    response = requests.post(url=LLM_URL, json=asdict(llm_parameters), headers=headers)
    return response.json()["choices"][0]["message"]["content"]
