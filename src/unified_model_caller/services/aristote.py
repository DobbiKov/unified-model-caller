from unified_model_caller.services.base import BaseService


class IlaasService(BaseService):
    def get_name(self) -> str:
        return "aristote"

    def requires_token(self) -> bool:
        return True

    def service_cooldown(self) -> int:
        return 0

    def call(self, model: str, prompt: str) -> str:
        import requests
        endpoint = "https://llm.aristote.education/v1/chat/completions"
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(endpoint, json=data, headers=headers)
        return response.json().get("choices")[0].get("message").get("content")
        
