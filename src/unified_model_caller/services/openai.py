from unified_model_caller.services.base import BaseService
from unified_model_caller.errors import ApiCallError


class OpenAIService(BaseService):
    def get_name(self) -> str:
        return "openai"

    def requires_token(self) -> bool:
        return True

    def service_cooldown(self) -> int:
        return 5000

    def call(self, model: str, prompt: str) -> str:
        import openai
        client = openai.OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        res = response.choices[0].message.content
        if res is None:
            raise ApiCallError(f"The call to the OpenAI's {model} model didn't provide any text result")
        return res
