from unified_model_caller.services.base import BaseService


class IlaasService(BaseService):
    def get_name(self) -> str:
        return "aristote"

    def requires_token(self) -> bool:
        return True

    def service_cooldown(self) -> int:
        return 0

    def call(self, model: str, prompt: str) -> str:
        from unified_model_caller.services._http import post_chat_completion
        return post_chat_completion(
            endpoint="https://llm.aristote.education/v1/chat/completions",
            model=model,
            prompt=prompt,
            service=self.get_name(),
            api_key=self.api_key,
        )
