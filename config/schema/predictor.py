from enum import Enum
from pydantic import model_validator
from config.schema.misc import LlmTypesEnum
from config.schema.base import ConfigBase


class PredictorEnum(str, Enum):
    llm = 'llm'

class PredictorLlmSettings(ConfigBase):
    type: LlmTypesEnum = LlmTypesEnum.openai
    name: str = 'gpt-3.5-turbo-0613'
    seed: int = 220
    num_workers: int = 5
    prompt: str = 'prompts/predictor_completion/prediction.prompt'
    mini_batch_size: int = 1
    mode: str = 'prediction'


class PredictorSettings(ConfigBase):
    method: PredictorEnum
    config: PredictorLlmSettings

    @model_validator(mode='after')
    def check_config_type(self):
        if self.method == PredictorEnum.llm and not isinstance(self.config, PredictorLlmSettings):
            raise TypeError(f"Config must be of type PredictorLlmSettings for method {self.method}")

        return self

