import requests


class LLM:
    def __init__(
        self,
        model: str,
        url: str,
        authorization: str,
        temperature: float,
        max_tokens: int,
    ):
        self.model = model
        self.url = url
        self.authorization = authorization
        self.temperature = temperature
        self.max_tokens = max_tokens

    def infer(self, system_prompt: str, user_prompt: str) -> str:
        headers = self._create_headers()
        parameters = self._create_parameters(system_prompt, user_prompt)

        print(f"Url used for LLM request: {self.url}")
        response = requests.post(url=self.url, headers=headers, json=parameters)

        return response.json()["choices"][0]["message"]["content"]

    def _create_headers(self) -> dict:
        headers = {}
        headers["Content-Type"] = "application/json"
        headers["Authorization"] = self.authorization
        return headers

    def _create_parameters(self, system_prompt: str, user_prompt: str) -> dict:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        parameters = {}
        parameters["model"] = self.model
        parameters["temperature"] = self.temperature
        parameters["max_tokens"] = self.max_tokens
        parameters["messages"] = messages

        return parameters
