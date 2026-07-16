from unified_model_caller.services.base import BaseService
from unified_model_caller.errors import (
    ApiCallError,
    ApiConnectionError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
)


class XAIService(BaseService):
    def get_name(self) -> str:
        return "xai"

    def requires_token(self) -> bool:
        return True

    def service_cooldown(self) -> int:
        return 5000

    def call(self, model: str, prompt: str) -> str:
        import grpc
        import xai_sdk
        from xai_sdk.chat import user as xai_user

        client = xai_sdk.Client(api_key=self.api_key)
        try:
            response = client.chat.create(
                model=model,
                messages=[xai_user(prompt)],
            ).sample()
        except grpc.RpcError as e:
            code = e.code()
            message = f"xAI API call failed ({code.name}): {e.details()}"
            grpc_error_map: dict[grpc.StatusCode, type[ApiCallError]] = {
                grpc.StatusCode.UNAUTHENTICATED: AuthenticationError,
                grpc.StatusCode.PERMISSION_DENIED: AuthenticationError,
                grpc.StatusCode.NOT_FOUND: NotFoundError,
                grpc.StatusCode.INVALID_ARGUMENT: BadRequestError,
                grpc.StatusCode.RESOURCE_EXHAUSTED: RateLimitError,
                grpc.StatusCode.UNAVAILABLE: ServiceUnavailableError,
                grpc.StatusCode.INTERNAL: ServiceUnavailableError,
                grpc.StatusCode.DEADLINE_EXCEEDED: ApiConnectionError,
            }
            error_cls = grpc_error_map.get(code, ApiCallError)
            raise error_cls(message, service=self.get_name()) from e
        return response.content
