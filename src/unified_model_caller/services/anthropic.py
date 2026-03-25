from unified_model_caller.services.base import BaseService
from unified_model_caller.errors import ApiCallError


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
        response = client.messages.create(
            max_tokens=10000,
            messages=messages,
            model=model,
        )
        text_resps = [resp for resp in response.content if resp.type == "text"]
        if len(text_resps) == 0:
            raise ApiCallError(f"The call to the Anthropic {model} model didn't provide any text response")
        return text_resps[0].text
