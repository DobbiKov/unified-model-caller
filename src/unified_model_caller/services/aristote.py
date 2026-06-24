from unified_model_caller.services.base import BaseService


class IlaasService(BaseService):
    def get_name(self) -> str:
        return "aristote"

    def requires_token(self) -> bool:
        return True

    def service_cooldown(self) -> int:
        return 0

    def call(self, model: str, prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://llm.aristote.education/v1",
            api_key=self.api_key
        )
         
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return resp.choices[0].text or ""
        
