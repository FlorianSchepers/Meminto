import os
import requests

def get_chat_completion(system_prompt, user_prompt, openai):

    json_data = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
    headers = {"Content-Type": "application/json"}

    if openai:
        url = "https://api.openai.com/v1/chat/completions"
        json_data["model"] = "gpt-3.5-turbo"
        headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
    else:
        url = os.environ["LLM_URL"]
        json_data["model"] = os.environ["LLM_MODEL"]
        json_data["max_tokens"] = os.environ["LLM_MAX_TOKENS"]
        headers ["Authorization"] = os.environ["LLM_AUTHORIZATION"]

    print(f"Url used for LLM request: {url}")
    response = requests.post(url=url, json=json_data, headers=headers)

    return response.json()["choices"][0]["message"]["content"]