from unified_model_caller.services.base import BaseService
from unified_model_caller.errors import ApiCallError, ModelOverloadedError


class GoogleService(BaseService):
    def get_name(self) -> str:
        return "google"

    def requires_token(self) -> bool:
        return True

    def service_cooldown(self) -> int:
        return 5000

    def call(self, model: str, prompt: str) -> str:
        from google import genai
        from google.genai import types as g_types
        from google.api_core import exceptions as google_exceptions

        client = genai.Client(api_key=self.api_key)

        try:
            contents = g_types.Content(
                role='user',
                parts=[g_types.Part.from_text(text=prompt)]
            )
            response = client.models.generate_content(
                model=model,
                contents=contents
            )
            return response.text or ""

        except google_exceptions.ResourceExhausted as e:
            raise ModelOverloadedError(f"Gemini model overloaded: {e}") from e
        except google_exceptions.TooManyRequests as e:
            error_msg = str(e)
            if "overloaded" in error_msg.lower():
                raise ModelOverloadedError(f"Gemini model overloaded: {error_msg}") from e
            raise ApiCallError(f"Gemini API call failed: {error_msg}") from e
        except Exception as e:
            error_msg = str(e)
            if "overloaded" in error_msg.lower():
                raise ModelOverloadedError(f"Gemini model overloaded: {error_msg}") from e
            print(f"Error communicating with Gemini API: {e}")
            raise ApiCallError(f"Gemini API call failed: {e}") from e
