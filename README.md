# Unified model caller

A small lightweight library for unified LLMs callings.

## Supported services 
- Aristote
- Google (gemini)
- Anthropic (claude)
- OpenAI (gpt)
- xAI (grok)

## Installation 
### Via `pip`:
```sh
pip install git+https://github.com/DobbiKov/unified-model-caller.git
```

### Via `uv`:
```sh
uv add git+https://github.com/DobbiKov/unified-model-caller.git
```

## Usage

```py
from unified_model_caller import LLMCaller

def main():
    # initialize caller
    caller = LLMCaller("google", "gemini-2.0-flash", "<your-token>")

    # make a call
    response = caller.call("What is a matrix?") 
    print(response)


if __name__ == "__main__":
    main()
```

## Roadmap
1. add ollama support
