from typing import Any, Callable, Set
from enum import Enum
from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    ImportString,
)
from pydantic.class_validators import model_validator

from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAIProviderSettings(BaseModel):
    api_key: str
    organization: str | None = None

class AzureOpenAIProviderSettings(BaseModel):
    api_key: str
    endpoint: str
    api_version: str

class DatasetSettings(BaseModel):
    name: str = 'dataset'
    records_path: str | None = None
    initial_dataset: str | None = None
    label_schema: list[str] = ["Yes", "No"]
    max_samples: int = 50
    semantic_sampling: bool = False # Change to True in case you don't have M1. Currently there is an issue with faiss and M1

class AnntatorEnum(str, Enum):
    argilla = 'argilla'
    llm = 'llm'

class LlmTypesEnum(str, Enum):
    openai = 'OpenAI'

class ArgillaSettings(BaseModel):
    api_url: str
    api_key: str
    workspace: str
    time_interval: int

class AnnotatorLlmSettings(BaseModel):
    type: LlmTypesEnum = LlmTypesEnum.openai
    name: str = 'gpt-4-1106-preview'
    instruction: str
    num_workers: int = 5
    prompt: str = 'prompts/predictor_completion/prediction.prompt'
    mini_batch_size: int = 1
    mode: str = 'annotation'


class AnnotatorSettings(BaseModel):
    method: AnntatorEnum
    config: ArgillaSettings | AnnotatorLlmSettings

    @model_validator
    def check_config_type(cls, values):
        method, config = values.get('method'), values.get('config')

        if method == AnntatorEnum.argilla and not isinstance(config, ArgillaSettings):
            raise TypeError(f"Config must be of type ArgillaSettings for method {method}")
        elif method == AnntatorEnum.llm and not isinstance(config, AnnotatorLlmSettings):
            raise TypeError(f"Config must be of type AnnotatorLlmSettings for method {method}")

        return values


class PredictorEnum(str, Enum):
    llm = 'llm'

class PredictorLlmSettings(BaseModel):
    type: LlmTypesEnum = LlmTypesEnum.openai
    name: str = 'gpt-3.5-turbo-0613'
    seed: int = 220
    num_workers: int = 5
    prompt: str = 'prompts/predictor_completion/prediction.prompt'
    mini_batch_size: int = 1
    mode: str = 'prediction'


class PredictorSettings(BaseModel):
    method: PredictorEnum
    config: AnnotatorLlmSettings

    @model_validator
    def check_config_type(cls, values):
        method, config = values.get('method'), values.get('config')

        if method == PredictorEnum.llm and not isinstance(config, PredictorLlmSettings):
            raise TypeError(f"Config must be of type PredictorLlmSettings for method {method}")

        return values


class MetaPromptsSettings(BaseModel):
    folder: str = 'prompts/meta_prompts_classification'
    num_err_prompt: int = 1 # Number of error examples per sample in the prompt generation
    num_err_samples: int = 2 # Number of error examples per sample in the sample generation
    history_length: int = 4 # Number of sample in the meta-prompt history
    num_generated_samples: int = 10 # Number of generated samples at each iteration
    num_initialize_samples: int = 10 # Number of generated samples at iteration 0, in zero-shot case
    samples_generation_batch: int = 10 # Number of samples generated in one call to the LLM
    num_workers: int = 5 #Number of parallel workers
    warmup: int = 4 # Number of warmup steps


class EvaluationSettings(BaseModel):
    function_name: str = 'accuracy'
    num_large_errors: int = 4
    num_boundary_predictions: int = 0
    error_threshold: float = 0.5


class StopCriteriaSettings(BaseModel):
    max_usage: int = 2 # In $ in case of OpenAI models, otherwise number of tokens
    patience: int = 10 # Number of patience steps
    min_delta: float = 0.01 # Delta for the improvement definition


class AnotherLlmSettings(BaseModel):
    type: LlmTypesEnum = LlmTypesEnum.openai
    name: str = 'gpt-4-1106-preview'
    temperature: float = 0.8

class Settings(BaseSettings):
    use_wandb: bool = True
    llm_provider: OpenAIProviderSettings | AzureOpenAIProviderSettings
    dataset: DatasetSettings
    annotator: AnnotatorSettings
    predictor: PredictorSettings
    meta_prompts: MetaPromptsSettings
    evaluation: EvaluationSettings
    stop_criteria: StopCriteriaSettings
    llm: AnotherLlmSettings

