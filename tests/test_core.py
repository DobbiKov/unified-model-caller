import pytest
from unittest.mock import patch, MagicMock

from unified_model_caller import LLMCaller, BaseService
from unified_model_caller.errors import InvalidServiceError
from unified_model_caller import core as core_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _EchoService(BaseService):
    def get_name(self) -> str:
        return "echo"

    def requires_token(self) -> bool:
        return False

    def service_cooldown(self) -> int:
        return 0

    def call(self, model: str, prompt: str) -> str:
        return f"{model}:{prompt}"


class _TokenService(BaseService):
    def get_name(self) -> str:
        return "tokenservice"

    def requires_token(self) -> bool:
        return True

    def service_cooldown(self) -> int:
        return 2000

    def call(self, model: str, prompt: str) -> str:
        return f"key={self.api_key}"


@pytest.fixture(autouse=True)
def _patch_services(monkeypatch):
    """Replace the global _SERVICES registry with a controlled set for every test."""
    fake = {
        "echo": _EchoService,
        "tokenservice": _TokenService,
    }
    monkeypatch.setattr(core_module, "_SERVICES", fake)
    yield fake


# ---------------------------------------------------------------------------
# LLMCaller.__init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_valid_service(self):
        caller = LLMCaller("echo", "model-x")
        assert caller.model == "model-x"

    def test_case_insensitive(self):
        caller = LLMCaller("ECHO", "model-x")
        assert isinstance(caller._service, _EchoService)

    def test_invalid_service_raises(self):
        with pytest.raises(InvalidServiceError):
            LLMCaller("nonexistent", "model-x")

    def test_api_key_passed_to_service(self):
        caller = LLMCaller("tokenservice", "m", api_key="secret")
        assert caller._service.api_key == "secret"

    def test_api_key_defaults_to_empty(self):
        caller = LLMCaller("echo", "m")
        assert caller._service.api_key == ""


# ---------------------------------------------------------------------------
# LLMCaller.call
# ---------------------------------------------------------------------------

class TestCall:
    def test_returns_service_response(self):
        caller = LLMCaller("echo", "gpt")
        assert caller.call("hello") == "gpt:hello"

    def test_api_key_available_inside_call(self):
        caller = LLMCaller("tokenservice", "m", api_key="tok123")
        assert caller.call("anything") == "key=tok123"

    def test_reraises_on_error(self):
        class _BrokenService(BaseService):
            def get_name(self): return "broken"
            def requires_token(self): return False
            def service_cooldown(self): return 0
            def call(self, model, prompt): raise RuntimeError("boom")

        core_module._SERVICES["broken"] = _BrokenService

        caller = LLMCaller("broken", "m")
        with pytest.raises(RuntimeError, match="boom"):
            caller.call("x")


# ---------------------------------------------------------------------------
# LLMCaller.wait_cooldown
# ---------------------------------------------------------------------------

class TestWaitCooldown:
    def test_sleeps_cooldown_in_seconds(self):
        caller = LLMCaller("tokenservice", "m")  # cooldown = 2000 ms
        with patch("unified_model_caller.core.time.sleep") as mock_sleep:
            caller.wait_cooldown()
            mock_sleep.assert_called_once_with(2.0)

    def test_zero_cooldown(self):
        caller = LLMCaller("echo", "m")  # cooldown = 0
        with patch("unified_model_caller.core.time.sleep") as mock_sleep:
            caller.wait_cooldown()
            mock_sleep.assert_called_once_with(0.0)


# ---------------------------------------------------------------------------
# LLMCaller.get_services
# ---------------------------------------------------------------------------

class TestGetServices:
    def test_returns_registered_names(self):
        services = LLMCaller.get_services()
        assert set(services) == {"echo", "tokenservice"}

    def test_returns_list(self):
        assert isinstance(LLMCaller.get_services(), list)


# ---------------------------------------------------------------------------
# LLMCaller.add_service
# ---------------------------------------------------------------------------

class TestAddService:
    def test_loads_and_registers_service(self, tmp_path):
        service_file = tmp_path / "myservice.py"
        service_file.write_text(
            "from unified_model_caller import BaseService\n"
            "class MyService(BaseService):\n"
            "    def get_name(self): return 'myservice'\n"
            "    def requires_token(self): return False\n"
            "    def service_cooldown(self): return 0\n"
            "    def call(self, model, prompt): return 'ok'\n"
        )
        LLMCaller.add_service(str(service_file))
        assert "myservice" in core_module._SERVICES

    def test_registered_service_is_callable(self, tmp_path):
        service_file = tmp_path / "dynservice.py"
        service_file.write_text(
            "from unified_model_caller import BaseService\n"
            "class DynService(BaseService):\n"
            "    def get_name(self): return 'dynservice'\n"
            "    def requires_token(self): return False\n"
            "    def service_cooldown(self): return 0\n"
            "    def call(self, model, prompt): return f'dyn:{prompt}'\n"
        )
        LLMCaller.add_service(str(service_file))
        caller = LLMCaller("dynservice", "m")
        assert caller.call("hi") == "dyn:hi"

    def test_nonexistent_file_raises(self):
        with pytest.raises((ValueError, FileNotFoundError, OSError)):
            LLMCaller.add_service("/nonexistent/path/service.py")

    def test_file_without_base_service_subclass_raises(self, tmp_path):
        service_file = tmp_path / "empty.py"
        service_file.write_text("x = 1\n")
        with pytest.raises(ValueError, match="No BaseService subclass"):
            LLMCaller.add_service(str(service_file))


# ---------------------------------------------------------------------------
# BaseService contract
# ---------------------------------------------------------------------------

class TestBaseService:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseService()  # type: ignore[abstract]

    def test_api_key_stored_on_init(self):
        svc = _EchoService("mykey")
        assert svc.api_key == "mykey"

    def test_default_api_key_empty(self):
        svc = _EchoService()
        assert svc.api_key == ""
