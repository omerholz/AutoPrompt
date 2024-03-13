from config.schema.base import ConfigBase
from pydantic import model_validator


class OpenAIProviderSettings(ConfigBase):
    api_key: str
    organization: str | None = None


class AzureOpenAIProviderSettings(ConfigBase):
    api_key: str
    endpoint: str
    api_version: str


class GoogleAIProviderSettings(ConfigBase):
    api_key: str


AnyProviderSettings = OpenAIProviderSettings | AzureOpenAIProviderSettings | GoogleAIProviderSettings


class Providers(ConfigBase):
    openai: OpenAIProviderSettings | AzureOpenAIProviderSettings | None = None
    google: GoogleAIProviderSettings | None = None

    @model_validator(mode='after')
    def check_provider(self):
        if self.openai is None and self.google is None:
            raise ValueError("At least one provider must be defined.")
        return self

