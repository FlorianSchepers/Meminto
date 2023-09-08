import os
import requests

def get_chat_completion(system_prompt, user_prompt):
    json_data = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
    headers = {"Content-Type": "application/json"}

    url = os.environ["LLM_URL"]
    json_data["model"] = os.environ["LLM_MODEL"]
    if os.environ["LLM_MODEL"] not in {"gpt-3.5-turbo", "gpt-4"}:
        json_data["max_tokens"] = os.environ["LLM_MAX_TOKENS"]
    headers ["Authorization"] = os.environ["LLM_AUTHORIZATION"]

    print(f"Url used for LLM request: {url}")
    response = requests.post(url=url, json=json_data, headers=headers)

    return response.json()["choices"][0]["message"]["content"]