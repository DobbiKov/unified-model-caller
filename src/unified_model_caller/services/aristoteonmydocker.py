from unified_model_caller.services.base import BaseService


class AristoteService(BaseService):
    def get_name(self) -> str:
        return "aristote-on-mydocker"

    def requires_token(self) -> bool:
        return False

    def service_cooldown(self) -> int:
        return 0

    def call(self, model: str, prompt: str) -> str:
        from unified_model_caller.services._http import post_chat_completion
        return post_chat_completion(
            endpoint="https://aristote-dispatcher.mydocker-run-vd.centralesupelec.fr/v1/chat/completions",
            model=model,
            prompt=prompt,
            service=self.get_name(),
        )
