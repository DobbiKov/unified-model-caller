from unified_model_caller.services.base import BaseService
from unified_model_caller.errors import ApiConnectionError, InvalidResponseError, error_from_status


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
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
        except openai.APIStatusError as e:
            raise error_from_status(e.status_code, f"OpenAI API call failed: {e}", service=self.get_name()) from e
        except openai.APIConnectionError as e:
            raise ApiConnectionError(f"Could not reach the OpenAI API: {e}", service=self.get_name()) from e
        res = response.choices[0].message.content
        if res is None:
            raise InvalidResponseError(
                f"The call to the OpenAI's {model} model didn't provide any text result",
                service=self.get_name(),
            )
        return res
