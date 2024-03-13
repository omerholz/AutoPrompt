from enum import Enum
from config.schema.base import ConfigBase


class LlmTypesEnum(str, Enum):
    openai = 'OpenAI'


class DatasetSettings(ConfigBase):
    name: str = 'dataset'
    records_path: str | None = None
    initial_dataset: str | None = None
    label_schema: list[str] = ["Yes", "No"]
    max_samples: int = 50
    semantic_sampling: bool = False # Change to True in case you don't have M1. Currently there is an issue with faiss and M1


class MetaPromptsSettings(ConfigBase):
    folder: str = 'prompts/meta_prompts_classification'
    num_err_prompt: int = 1 # Number of error examples per sample in the prompt generation
    num_err_samples: int = 2 # Number of error examples per sample in the sample generation
    history_length: int = 4 # Number of sample in the meta-prompt history
    num_generated_samples: int = 10 # Number of generated samples at each iteration
    num_initialize_samples: int = 10 # Number of generated samples at iteration 0, in zero-shot case
    samples_generation_batch: int = 10 # Number of samples generated in one call to the LLM
    num_workers: int = 5 #Number of parallel workers
    warmup: int = 4 # Number of warmup steps


class EvaluationSettings(ConfigBase):
    function_name: str = 'accuracy'
    num_large_errors: int = 4
    num_boundary_predictions: int = 0
    error_threshold: float = 0.5


class StopCriteriaSettings(ConfigBase):
    max_usage: int = 2 # In $ in case of OpenAI models, otherwise number of tokens
    patience: int = 10 # Number of patience steps
    min_delta: float = 0.01 # Delta for the improvement definition


class AnotherLlmSettings(ConfigBase):
    type: LlmTypesEnum = LlmTypesEnum.openai
    name: str = 'gpt-4-1106-preview'
    temperature: float = 0.8

class OpenAI:
    pass
