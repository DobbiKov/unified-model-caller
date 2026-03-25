from abc import ABC, abstractmethod


class BaseService(ABC):
    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    @abstractmethod
    def call(self, model: str, prompt: str) -> str: ...

    @abstractmethod
    def requires_token(self) -> bool: ...

    @abstractmethod
    def get_name(self) -> str: ...

    @abstractmethod
    def service_cooldown(self) -> int: ...
