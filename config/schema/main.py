from config.schema.base import ConfigBase
from config.schema.providers import Providers
from config.schema.misc import DatasetSettings
from config.schema.annotator import AnnotatorSettings
from config.schema.predictor import PredictorSettings

class Settings(ConfigBase):
    use_wandb: bool = True
    providers: Providers
    dataset: DatasetSettings
    annotator: AnnotatorSettings
    predictor: PredictorSettings
    # meta_prompts: MetaPromptsSettings
    # evaluation: EvaluationSettings
    # stop_criteria: StopCriteriaSettings
    # llm: AnotherLlmSettings
    # old_config = SettingsConfigDict(yaml_file='config_default.yml')
    # old_llm_env = SettingsConfigDict(yaml_file='llm_env.yml')

