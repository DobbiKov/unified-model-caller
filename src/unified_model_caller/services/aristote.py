from unified_model_caller.services.base import BaseService


class AristoteService(BaseService):
    def get_name(self) -> str:
        return "aristoteonmydocker"

    def requires_token(self) -> bool:
        return False

    def service_cooldown(self) -> int:
        return 0

    def call(self, model: str, prompt: str) -> str:
        import requests
        endpoint = "https://aristote-dispatcher.mydocker-run-vd.centralesupelec.fr/v1/chat/completions"
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = requests.post(endpoint, json=data)
        return response.json().get("choices")[0].get("message").get("content")
