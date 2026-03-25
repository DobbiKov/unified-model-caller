# Unified Model Caller

A small, lightweight library that provides a single unified interface for
calling LLMs from different providers. Instead of learning each provider's SDK
separately, you instantiate one `LLMCaller` and swap the service name.

## Supported services

| Service name | Provider |
|---|---|
| `openai` | OpenAI (GPT models) |
| `anthropic` | Anthropic (Claude models) |
| `google` | Google (Gemini models) |
| `xai` | xAI (Grok models) |
| `ilaas` | Ilaas |
| `aristoteonmydocker` | Aristote on MyDocker |

## Installation

**Via pip:**
```sh
pip install git+https://github.com/DobbiKov/unified-model-caller.git
```

**Via uv:**
```sh
uv add git+https://github.com/DobbiKov/unified-model-caller.git
```

## Usage

```python
from unified_model_caller import LLMCaller

caller = LLMCaller("google", "gemini-2.0-flash", api_key="<your-api-key>")
response = caller.call("What is a matrix?")
print(response)
```

The constructor signature is:

```python
LLMCaller(service: str, model: str, api_key: str = "")
```

- `service` — case-insensitive service name (see table above)
- `model` — model identifier string passed directly to the provider
- `api_key` — API key; can be omitted for services that don't require one

### Rate limiting

Call `wait_cooldown()` between requests to respect each service's built-in cooldown:

```python
caller.wait_cooldown()
response = caller.call("Next prompt")
```

### Listing available services

```python
LLMCaller.get_services()
# ['openai', 'anthropic', 'google', 'xai', 'ilaas', 'aristoteonmydocker']
```

## Adding an external service

You can register a new service at runtime from any Python file — no changes to the library are needed.

### 1. Create a service file

The file must define a class that inherits from `BaseService` and implements four methods:

```python
# my_service.py
from unified_model_caller import BaseService

class MyService(BaseService):
    def get_name(self) -> str:
        """Unique lowercase name used to identify this service."""
        return "myservice"

    def requires_token(self) -> bool:
        """Return True if the service needs an API key."""
        return True

    def service_cooldown(self) -> int:
        """Minimum delay between calls, in milliseconds."""
        return 1000

    def call(self, model: str, prompt: str) -> str:
        """Send prompt to the model and return the response text."""
        import requests
        response = requests.post(
            "https://api.myservice.example/v1/completions",
            json={"model": model, "prompt": prompt},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        return response.json()["text"]
```

The `api_key` passed to `LLMCaller(...)` is available as `self.api_key` inside your class.

### 2. Register and use it

```python
from unified_model_caller import LLMCaller

LLMCaller.add_service("/path/to/my_service.py")

caller = LLMCaller("myservice", "my-model-name", api_key="<your-api-key>")
response = caller.call("Hello!")
print(response)
```

`add_service` loads the file, finds the `BaseService` subclass inside it, and registers it globally under the name returned by `get_name()`. The service is then available to all subsequent `LLMCaller` instances in the same process.

### BaseService contract

| Method | Return type | Description |
|---|---|---|
| `get_name(self)` | `str` | Unique service identifier (lowercase). Used as the `service` argument to `LLMCaller`. |
| `requires_token(self)` | `bool` | Whether the service needs an API key. |
| `service_cooldown(self)` | `int` | Cooldown between calls in milliseconds. |
| `call(self, model, prompt)` | `str` | Perform the API call and return the response text. |
