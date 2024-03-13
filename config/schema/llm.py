from enum import Enum
from config.schema.base import ConfigBase
from config.schema.providers import AnyProviderSettings, OpenAIProviderSettings, AzureOpenAIProviderSettings, \
    GoogleAIProviderSettings
from pydantic import model_validator, Field


class LlmTypesEnum(str, Enum):
    openai = 'OpenAI'

    @classmethod # make case-insensitive Enum
    def _missing_(cls, value):
        value = value.lower()
        for member in cls:
            if member.value.lower() == value:
                return member
        return None


class ChatModelOpenAISettings(ConfigBase):
    name: str = Field(alias='model_name')
    temperature: float = 0.8
    model_kwargs: dict | None = None

    class Config:
        populate_by_name = True


class ChatModelAzureOpenAISettings(ConfigBase):
    name: str = Field(alias='deployment_name')
    temperature: float = 0.8
    model_kwargs: dict | None = None

    class Config:
        populate_by_name = True


class ChatModelGoogleAISettings(ConfigBase):
    name: str = Field(alias='model')
    temperature: float = 0.8
    model_kwargs: dict | None = None

    class Config:
        populate_by_name = True


AnyChatModelSettings = ChatModelOpenAISettings | ChatModelAzureOpenAISettings | ChatModelGoogleAISettings


class LlmSettings(ConfigBase):
    provider: AnyProviderSettings
    model: AnyChatModelSettings

    @model_validator(mode='after')
    def check_provider_match_model(self):
        if isinstance(self.model, ChatModelOpenAISettings) and not isinstance(self.provider, OpenAIProviderSettings):
            raise TypeError(f"Provider must be of type OpenAIProviderSettings for model {type(self.model)}")
        elif isinstance(self.model, ChatModelAzureOpenAISettings) and not isinstance(self.provider,
                                                                                     AzureOpenAIProviderSettings):
            raise TypeError(f"Provider must be of type AzureOpenAIProviderSettings for model {type(self.model)}")
        elif isinstance(self.model, ChatModelGoogleAISettings) and not isinstance(self.provider,
                                                                                  GoogleAIProviderSettings):
            raise TypeError(f"Provider must be of type GoogleAIProviderSettings for model {type(self.model)}")
        return self

    def args_dump(self, *args, **kwargs):
        return {
            k: v
            for k, v in self.provider.clean_model_dump(*args, **kwargs)
        }.update({
            k: v
            for k, v in self.model.clean_model_dump(*args, **kwargs)
        })


class PredictorLlmSettings(ConfigBase):
    type: LlmTypesEnum = LlmTypesEnum.openai
    name: str = 'gpt-3.5-turbo-0613'
    seed: int = 220
    num_workers: int = 5
    prompt: str = 'prompts/predictor_completion/prediction.prompt'
    mini_batch_size: int = 1
    mode: str = 'prediction'


class AnnotatorLlmSettings(ConfigBase):
    type: LlmTypesEnum = LlmTypesEnum.openai
    name: str = 'gpt-4-1106-preview'
    instruction: str
    num_workers: int = 5
    prompt: str = 'prompts/predictor_completion/prediction.prompt'
    mini_batch_size: int = 1
    mode: str = 'annotation'


class AnotherLlmSettings(ConfigBase):
    type: LlmTypesEnum = LlmTypesEnum.openai
    name: str = 'gpt-4-1106-preview'
    temperature: float = 0.8
