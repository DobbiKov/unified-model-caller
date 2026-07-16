from unified_model_caller.services.base import BaseService
from unified_model_caller.errors import ApiConnectionError, InvalidResponseError, error_from_status


class AnthropicService(BaseService):
    def get_name(self) -> str:
        return "anthropic"

    def requires_token(self) -> bool:
        return True

    def service_cooldown(self) -> int:
        return 5000

    def call(self, model: str, prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        messages = [anthropic.types.MessageParam(content=prompt, role='user')]
        try:
            response = client.messages.create(
                max_tokens=10000,
                messages=messages,
                model=model,
            )
        except anthropic.APIStatusError as e:
            raise error_from_status(e.status_code, f"Anthropic API call failed: {e}", service=self.get_name()) from e
        except anthropic.APIConnectionError as e:
            raise ApiConnectionError(f"Could not reach the Anthropic API: {e}", service=self.get_name()) from e
        text_resps = [resp for resp in response.content if resp.type == "text"]
        if len(text_resps) == 0:
            raise InvalidResponseError(
                f"The call to the Anthropic {model} model didn't provide any text response",
                service=self.get_name(),
            )
        return text_resps[0].text
