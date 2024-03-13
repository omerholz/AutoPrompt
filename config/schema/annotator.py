from enum import Enum
from pydantic import model_validator
from config.schema.base import ConfigBase
from config.schema.misc import LlmTypesEnum
from config.schema.providers import AnyProviderSettings
from config.schema.llm import LlmSettings

class AnntatorEnum(str, Enum):
    argilla = 'argilla'
    llm = 'llm'
    llm_batch = 'llm_batch'


class ArgillaSettings(ConfigBase):
    api_url: str
    api_key: str
    workspace: str
    time_interval: int


class AnnotatorLlmSettings(ConfigBase):
    llm_settings: LlmSettings
    instruction: str
    num_workers: int = 5
    prompt: str = 'prompts/predictor_completion/prediction.prompt'
    mini_batch_size: int = 1
    mode: str = 'annotation'


class AnnotatorSettings(ConfigBase):
    method: AnntatorEnum
    config: ArgillaSettings | AnnotatorLlmSettings

    @model_validator(mode='after')
    def check_config_type(self):

        if self.method == AnntatorEnum.argilla and not isinstance(self.config, ArgillaSettings):
            raise TypeError(f"Config must be of type ArgillaSettings for method {self.method}")
        elif self.method == AnntatorEnum.llm and not isinstance(self.config, AnnotatorLlmSettings):
            raise TypeError(f"Config must be of type AnnotatorLlmSettings for method {self.method}")

        return self

