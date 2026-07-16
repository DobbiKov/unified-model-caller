from unified_model_caller.services.base import BaseService
from unified_model_caller.errors import (
    ApiCallError,
    ApiConnectionError,
    ModelOverloadedError,
    error_from_status,
)


class GoogleService(BaseService):
    def get_name(self) -> str:
        return "google"

    def requires_token(self) -> bool:
        return True

    def service_cooldown(self) -> int:
        return 5000

    def call(self, model: str, prompt: str) -> str:
        from google import genai
        from google.genai import errors as g_errors
        from google.genai import types as g_types

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

        except g_errors.APIError as e:
            error_msg = f"Gemini API call failed: {e}"
            # Gemini reports overload as 429 RESOURCE_EXHAUSTED or 503 UNAVAILABLE
            if "overload" in str(e).lower():
                raise ModelOverloadedError(
                    f"Gemini model overloaded: {e}", service=self.get_name(), status_code=e.code
                ) from e
            if isinstance(e.code, int):
                raise error_from_status(e.code, error_msg, service=self.get_name()) from e
            raise ApiCallError(error_msg, service=self.get_name()) from e
        except ConnectionError as e:
            raise ApiConnectionError(f"Could not reach the Gemini API: {e}", service=self.get_name()) from e
        except Exception as e:
            error_msg = str(e)
            if "overload" in error_msg.lower():
                raise ModelOverloadedError(f"Gemini model overloaded: {error_msg}", service=self.get_name()) from e
            raise ApiCallError(f"Gemini API call failed: {e}", service=self.get_name()) from e
