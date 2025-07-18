import os
import requests
import openai
import anthropic
from google import genai
from google.genai import types as g_types


from unified_model_caller.enums import Service
from unified_model_caller.errors import ApiCallError

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

    def __init__(self, service: str, model: str):
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

    def _dispatch(self, prompt: str) -> str:
        """Route *services[idx]* to the appropriate caller."""
        handler = self._handlers.get(self.service)
        if handler is None:
            raise RuntimeError(f"Didn't find an appropriate handler for the {self.service} service")
        return handler(self, prompt)

    @_handler([Service.OpenAI])
    def _call_openai(self, prompt: str) -> str:
        """Handles the API call to OpenAI."""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        client = openai.OpenAI()
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
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")

        client = anthropic.Anthropic()
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
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")

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
        
        except Exception as e:
            print(f"Error communicating with Gemini API: {e}")
            raise ApiCallError(f"Gemini API call failed: {e}")



    @_handler([Service.Aristote])
    def _call_aristote(self, prompt: str) -> str:
        """Handles the API call to Artistote models"""
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
    def _call_unsupported(self, prompt: str) -> str:
        """Handles calls to services that are not yet implemented."""
        raise NotImplementedError(
            f"The service '{self.service}' is recognized but not yet implemented."
        )


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
