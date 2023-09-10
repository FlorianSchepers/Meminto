import os
import requests

LLM_URL = os.environ["LLM_URL"]
LLM_MODEL = os.environ["LLM_MODEL"]
LLM_MAX_TOKENS = os.environ["LLM_MAX_TOKENS"] if "LLM_MAX_TOKENS" in os.environ else "4000"
LLM_AUTHORIZATION = os.environ["LLM_AUTHORIZATION"]

def infer_llm(system_prompt, user_prompt):
    json_data = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "model": LLM_MODEL,
    }
    if LLM_MODEL not in {"gpt-3.5-turbo", "gpt-4"}:
        json_data["max_tokens"] = LLM_MAX_TOKENS

    headers = {"Content-Type": "application/json", "Authorization": LLM_AUTHORIZATION}

    print(f"Url used for LLM request: {LLM_URL}")
    response = requests.post(url=LLM_URL, json=json_data, headers=headers)

    return response.json()["choices"][0]["message"]["content"]
