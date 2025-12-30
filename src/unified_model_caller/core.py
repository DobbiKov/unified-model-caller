import time


from unified_model_caller.enums import Service, service_cooldown
from unified_model_caller.errors import ApiCallError, ModelOverloadedError

def _collect_handlers(cls):
    cls._handlers = {}
    for name, member in cls.__dict__.items():
        services = getattr(member, "_services", None)
        if services is not None:
            for service in services:
                cls._handlers[service] = member
    return cls

def _handler(services: list[Service]):
    def decorator(fn):
        fn._services = services 
        return fn
    return decorator

@_collect_handlers
class LLMCaller:
    """
    A unified caller for various Large Language Model (LLM) APIs.

    This class provides a single interface to call different LLM providers
    like OpenAI, Anthropic, and Google. It abstracts away the specific API
    details of each service.

    Attributes:
        service (str): The name of the LLM service (e.g., 'openai').
        model (str): The specific model name for the service.
    """
    # _handlers: dict[Service, Callable[[str], str]] = {}
    _handlers = {}

    def __init__(self, service: str, model: str, api_key: str = ""):
        """
        Initializes the LLMCaller.

        Args:
            service (str): The LLM service to use.
                           Supported services: 'openai', 'anthropic', 'google'.
                           Experimental: 'xai', 'aristote'.
            model (str): The model to use for the specified service.
        """
        self.service = Service.from_str(service.lower())
        self.model = model
        self.api_key = api_key

    def _dispatch(self, prompt: str) -> str:
        """Route *services[idx]* to the appropriate caller."""
        handler = self._handlers.get(self.service)
        if handler is None:
            raise RuntimeError(f"Didn't find an appropriate handler for the {self.service} service")
        return handler(self, prompt)

    @_handler([Service.OpenAI])
    def _call_openai(self, prompt: str) -> str:
        """Handles the API call to OpenAI."""
        import openai
        client = openai.OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        res = response.choices[0].message.content
        if res is None:
            raise ApiCallError(f"The call to the OpenAI's {self.model} model didn't provide any text result")
        return res

    @_handler([Service.Anthropic])
    def _call_anthropic(self, prompt: str) -> str:
        """Handles the API call to Anthropic."""
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        messages = [anthropic.types.MessageParam(content=prompt, role='user')]
        response = client.messages.create(
            max_tokens=10000,
            messages=messages,
            model=self.model,
        )
        text_resps = [resp for resp in response.content if resp.type == "text"]
        if len(text_resps) == 0:
            raise ApiCallError(f"The call to the Anthropic {self.model} model didn't provide any text response")
        return text_resps[0].text

    @_handler([Service.Google])
    def _call_google(self, prompt: str) -> str:
        """Handles the API call to Google's Generative AI."""
        from google import genai
        from google.genai import types as g_types
        from google.api_core import exceptions as google_exceptions
        api_key = self.api_key

        client = genai.Client(api_key=api_key)

        try:
            contents = g_types.Content(
                    role='user',
                    parts=[g_types.Part.from_text(text=prompt)]
            )
            

            # await asyncio.sleep(INTER_FILE_TRANSLATION_DELAY_SECONDS)
            response = client.models.generate_content(
                    model=self.model,
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



    @_handler([Service.AristoteOnMyDocker])
    def _call_aristote(self, prompt: str) -> str:
        """Handles the API call to Artistote models"""
        import requests
        aristote_API_ENDPOINT = "https://aristote-dispatcher.mydocker-run-vd.centralesupelec.fr/v1/chat/completions"
        model = self.model
        # default model: "casperhansen/llama-3.3-70b-instruct-awq" 
        data = {
            "model": model, 
            "messages": [{"role": "user", "content":prompt}],  
        }
        response = requests.post(aristote_API_ENDPOINT, json=data)
        return response.json().get("choices")[0].get("message").get("content")


    @_handler([Service.xAI])
    def _call_xai(self, prompt: str) -> str:
        import xai_sdk
        from xai_sdk.chat import user as xai_user
        client = xai_sdk.Client(api_key=self.api_key)

        response = client.chat.create(
            model=self.model,
            messages=[
                xai_user(prompt)
            ],
        ).sample()
        return response.content

    def _call_unsupported(self, prompt: str) -> str:
        """Handles calls to services that are not yet implemented."""
        raise NotImplementedError(
            f"The service '{self.service}' is recognized but not yet implemented."
        )

    def wait_cooldown(self) -> None:
        """
        Waits an amount of time required by service to respect the limits.
        """
        cooldown = service_cooldown[self.service]
        time.sleep(cooldown/1000) # /1000 because sleep takes seconds

    def call(self, prompt: str) -> str:
        """
        Sends a prompt to the configured LLM and returns the response.

        Args:
            prompt (str): The input text to send to the model.

        Returns:
            str: The text generated by the model in response to the prompt.
        """
        try:
            return self._dispatch(prompt)
        except Exception as e:
            # Log the exception or handle it as needed
            print(f"An error occurred while calling the {self.service} API: {e}")
            raise e
