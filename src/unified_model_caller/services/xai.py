from unified_model_caller.services.base import BaseService


class XAIService(BaseService):
    def get_name(self) -> str:
        return "xai"

    def requires_token(self) -> bool:
        return True

    def service_cooldown(self) -> int:
        return 5000

    def call(self, model: str, prompt: str) -> str:
        import xai_sdk
        from xai_sdk.chat import user as xai_user
        client = xai_sdk.Client(api_key=self.api_key)
        response = client.chat.create(
            model=model,
            messages=[xai_user(prompt)],
        ).sample()
        return response.content
