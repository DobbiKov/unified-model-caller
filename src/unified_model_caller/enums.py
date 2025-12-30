import enum

from unified_model_caller.errors import InvalidServiceError

class Service(str, enum.Enum):
    """Enumeration for supported languages."""
    AristoteOnMyDocker = "AristoteOnMyDocker"
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
    def requires_token(self) -> bool:
        """
        Returns True if the service requires token and False otherwise.
        """

        match self:
            case Service.AristoteOnMyDocker:
                return False
            case _:
                return True

service_cooldown = { # in ms
        Service.AristoteOnMyDocker:   0,
        Service.Anthropic:  5000,
        Service.OpenAI:     5000,
        Service.Google:     5000,
        Service.xAI:        5000
}
