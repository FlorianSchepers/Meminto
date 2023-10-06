import os
import requests

LLM_URL = os.environ["LLM_URL"]
LLM_MODEL = os.environ["LLM_MODEL"]
LLM_AUTHORIZATION = os.environ["LLM_AUTHORIZATION"]
LLM_MAX_TOKENS = os.environ["LLM_MAX_TOKENS"] if "LLM_MAX_TOKENS" in os.environ else "4000"

def get_json_data(system_prompt, user_prompt):
    json_data = {}

    json_data["model"] = LLM_MODEL
    json_data["temperature"] = 0.001
    json_data["max_tokens"] = int(LLM_MAX_TOKENS)

    if LLM_MODEL == "Llama-2-70b-chat-hf":
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