import os
import requests

LLM_URL = os.environ["LLM_URL"]
LLM_MODEL = os.environ["LLM_MODEL"]
LLM_AUTHORIZATION = os.environ["LLM_AUTHORIZATION"]
LLM_MAX_TOKENS = os.environ["LLM_MAX_TOKENS"] if "LLM_MAX_TOKENS" in os.environ else "4000"
LLM_TEMPERATURE = os.environ["LLM_TEMPERATURE"] if "LLM_TEMPERATURE" in os.environ else "0.6"

def get_json_data(system_prompt, user_prompt):
    json_data = {}

    json_data["model"] = LLM_MODEL
    json_data["temperature"] = LLM_TEMPERATURE

    if LLM_MODEL not in {"gpt-3.5-turbo", "gpt-4"}:
        json_data["max_tokens"] = LLM_MAX_TOKENS

    if LLM_MODEL == "llama2-chat":
        system_prompt = "<s>[INST] <<SYS>>\n" + system_prompt + "\n<</SYS>>\n\n"
        user_prompt = user_prompt + "[/INST]"

    json_data["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    return json_data

def get_headers():
    headers = {}
    headers["Content-Type"] = "application/json"
    headers["Authorization"] = LLM_AUTHORIZATION
    return headers

def infer_llm(system_prompt, user_prompt):
    json_data = get_json_data(system_prompt, user_prompt)
    headers = get_headers()
    
    print(f"Url used for LLM request: {LLM_URL}")
    response = requests.post(url=LLM_URL, json=json_data, headers=headers)

    return response.json()["choices"][0]["message"]["content"]
