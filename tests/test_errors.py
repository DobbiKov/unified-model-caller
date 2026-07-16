import pytest
from unittest.mock import MagicMock, patch

from unified_model_caller.errors import (
    ApiCallError,
    ApiConnectionError,
    AuthenticationError,
    BadRequestError,
    InvalidModelError,
    InvalidResponseError,
    InvalidServiceError,
    ModelOverloadedError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    UnifiedModelCallerError,
    error_from_status,
)
from unified_model_caller.services._http import post_chat_completion


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------

class TestHierarchy:
    def test_all_errors_share_a_common_base(self):
        for error_cls in (
            InvalidServiceError, InvalidModelError, ApiCallError,
            AuthenticationError, BadRequestError, NotFoundError,
            RateLimitError, ModelOverloadedError, ServiceUnavailableError,
            ApiConnectionError, InvalidResponseError,
        ):
            assert issubclass(error_cls, UnifiedModelCallerError)

    def test_http_errors_are_api_call_errors(self):
        for error_cls in (
            AuthenticationError, BadRequestError, NotFoundError,
            RateLimitError, ModelOverloadedError, ServiceUnavailableError,
            ApiConnectionError, InvalidResponseError,
        ):
            assert issubclass(error_cls, ApiCallError)

    def test_api_call_error_carries_context(self):
        err = AuthenticationError("bad key", service="ilaas", status_code=401)
        assert err.service == "ilaas"
        assert err.status_code == 401
        assert str(err) == "bad key"

    def test_context_defaults_to_none(self):
        err = ApiCallError("boom")
        assert err.service is None
        assert err.status_code is None


# ---------------------------------------------------------------------------
# error_from_status
# ---------------------------------------------------------------------------

class TestErrorFromStatus:
    @pytest.mark.parametrize("status,expected", [
        (400, BadRequestError),
        (401, AuthenticationError),
        (403, AuthenticationError),
        (404, NotFoundError),
        (422, BadRequestError),
        (429, RateLimitError),
        (500, ServiceUnavailableError),
        (502, ServiceUnavailableError),
        (503, ServiceUnavailableError),
        (529, ModelOverloadedError),
        (418, ApiCallError),
    ])
    def test_status_code_mapping(self, status, expected):
        err = error_from_status(status, "msg", service="svc")
        assert type(err) is expected
        assert err.status_code == status
        assert err.service == "svc"

    def test_5xx_mentioning_overload_maps_to_overloaded(self):
        err = error_from_status(503, "the model is Overloaded, retry later")
        assert type(err) is ModelOverloadedError


# ---------------------------------------------------------------------------
# post_chat_completion (requests-based services)
# ---------------------------------------------------------------------------

def _mock_response(status_code=200, json_data=None, text=""):
    response = MagicMock()
    response.status_code = status_code
    response.ok = status_code < 400
    response.text = text
    if json_data is not None:
        response.json.return_value = json_data
    else:
        response.json.side_effect = ValueError("no json")
    return response


class TestPostChatCompletion:
    def _call(self, response=None, side_effect=None):
        with patch("unified_model_caller.services._http.requests.post") as mock_post:
            if side_effect is not None:
                mock_post.side_effect = side_effect
            else:
                mock_post.return_value = response
            return post_chat_completion(
                endpoint="https://example.test/v1/chat/completions",
                model="m",
                prompt="hi",
                service="ilaas",
                api_key="key",
            )

    def test_returns_content_on_success(self):
        response = _mock_response(json_data={
            "choices": [{"message": {"content": "hello"}}]
        })
        assert self._call(response) == "hello"

    @pytest.mark.parametrize("status,expected", [
        (401, AuthenticationError),
        (403, AuthenticationError),
        (404, NotFoundError),
        (429, RateLimitError),
        (500, ServiceUnavailableError),
    ])
    def test_http_error_statuses(self, status, expected):
        response = _mock_response(status_code=status, text="details")
        with pytest.raises(expected) as excinfo:
            self._call(response)
        assert excinfo.value.status_code == status
        assert excinfo.value.service == "ilaas"

    def test_network_failure_raises_connection_error(self):
        import requests
        with pytest.raises(ApiConnectionError):
            self._call(side_effect=requests.ConnectionError("refused"))

    def test_timeout_raises_connection_error(self):
        import requests
        with pytest.raises(ApiConnectionError):
            self._call(side_effect=requests.Timeout("too slow"))

    def test_malformed_body_raises_invalid_response(self):
        response = _mock_response(json_data={"unexpected": "shape"})
        with pytest.raises(InvalidResponseError):
            self._call(response)

    def test_non_json_body_raises_invalid_response(self):
        response = _mock_response(text="<html>gateway</html>")
        with pytest.raises(InvalidResponseError):
            self._call(response)

    def test_null_content_raises_invalid_response(self):
        response = _mock_response(json_data={
            "choices": [{"message": {"content": None}}]
        })
        with pytest.raises(InvalidResponseError):
            self._call(response)
