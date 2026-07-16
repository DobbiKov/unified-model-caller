"""Exception types raised by unified_model_caller.

Hierarchy:

    UnifiedModelCallerError
    ├── InvalidServiceError
    ├── InvalidModelError
    └── ApiCallError
        ├── AuthenticationError      (HTTP 401 / 403)
        ├── BadRequestError          (HTTP 400 / 422)
        ├── NotFoundError            (HTTP 404)
        ├── RateLimitError           (HTTP 429)
        ├── ModelOverloadedError     (HTTP 529, or 5xx reporting overload)
        ├── ServiceUnavailableError  (other HTTP 5xx)
        ├── ApiConnectionError       (network failure / timeout)
        └── InvalidResponseError     (empty or malformed response body)

Catching `ApiCallError` handles every failure of an API call; catching
`UnifiedModelCallerError` handles everything the library can raise.
"""


class UnifiedModelCallerError(Exception):
    """Base class for all errors raised by this library."""


class InvalidServiceError(UnifiedModelCallerError):
    """Error related to the invalid choice of the service"""


class InvalidModelError(UnifiedModelCallerError):
    """Error related to the invalid choice of the model"""


class ApiCallError(UnifiedModelCallerError):
    """Error related to the calls to models via API.

    Attributes:
        service: Name of the service the failing call was made through, if known.
        status_code: HTTP status code, when the failure came from an HTTP response.
    """

    def __init__(self, message: str, *, service: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.service = service
        self.status_code = status_code


class AuthenticationError(ApiCallError):
    """The API key is missing, invalid, or lacks permission (HTTP 401/403)."""


class BadRequestError(ApiCallError):
    """The provider rejected the request as malformed (HTTP 400/422)."""


class NotFoundError(ApiCallError):
    """The requested resource (usually the model) does not exist (HTTP 404)."""


class RateLimitError(ApiCallError):
    """The provider is rate-limiting the caller (HTTP 429)."""


class ModelOverloadedError(ApiCallError):
    """Error raised when a model reports being overloaded"""


class ServiceUnavailableError(ApiCallError):
    """The provider had an internal error or is down (HTTP 5xx)."""


class ApiConnectionError(ApiCallError):
    """The provider could not be reached (network failure or timeout)."""


class InvalidResponseError(ApiCallError):
    """The provider answered, but the response was empty or malformed."""


def error_from_status(status_code: int, message: str, service: str | None = None) -> ApiCallError:
    """Builds the ApiCallError subclass matching an HTTP status code.

    Returns the error instead of raising it, so callers can chain the original
    exception: ``raise error_from_status(...) from original``.
    """
    error_cls: type[ApiCallError]
    if status_code in (401, 403):
        error_cls = AuthenticationError
    elif status_code == 404:
        error_cls = NotFoundError
    elif status_code == 429:
        error_cls = RateLimitError
    elif status_code in (400, 422):
        error_cls = BadRequestError
    elif status_code == 529 or (status_code >= 500 and "overload" in message.lower()):
        error_cls = ModelOverloadedError
    elif status_code >= 500:
        error_cls = ServiceUnavailableError
    else:
        error_cls = ApiCallError
    return error_cls(message, service=service, status_code=status_code)
