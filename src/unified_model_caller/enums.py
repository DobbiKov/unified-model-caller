import enum

from unified_model_caller.errors import InvalidServiceError

class Service(str, enum.Enum):
    """Enumeration for supported languages."""
    Aristote = "Aristote"
    Anthropic = "Anthropic"
    OpenAI = "OpenAI"
    Google = "Google"
    xAI = "xAI"

    @classmethod
    def from_str(cls, s: str) -> 'Service':
        for lang_member in cls:
            if lang_member.value.lower() == s.lower():
                return lang_member
        raise InvalidServiceError(f"'{s}' is not a valid service")

    def __str__(self) -> str:
        return self.value

