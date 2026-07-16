"""Shared helper for services that call an OpenAI-compatible HTTP endpoint directly."""

import requests

from unified_model_caller.errors import (
    ApiConnectionError,
    InvalidResponseError,
    error_from_status,
)


def post_chat_completion(
    endpoint: str,
    model: str,
    prompt: str,
    service: str,
    api_key: str = "",
    timeout: float = 120,
) -> str:
    """POSTs a chat-completion request and returns the response text.

    Raises the ApiCallError subclass matching the failure: ApiConnectionError
    when the endpoint cannot be reached, the status-code-specific error for
    non-2xx responses, and InvalidResponseError for unusable response bodies.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        response = requests.post(endpoint, json=data, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        raise ApiConnectionError(f"Could not reach the {service} API: {e}", service=service) from e

    if not response.ok:
        detail = response.text.strip()[:500] or response.reason
        raise error_from_status(
            response.status_code,
            f"The {service} API returned HTTP {response.status_code}: {detail}",
            service=service,
        )

    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise InvalidResponseError(
            f"Unexpected response format from the {service} API: {response.text.strip()[:500]}",
            service=service,
            status_code=response.status_code,
        ) from e
    if content is None:
        raise InvalidResponseError(
            f"The {service} API response contained no text content",
            service=service,
            status_code=response.status_code,
        )
    return content
