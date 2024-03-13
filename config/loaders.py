
from dynaconf import Dynaconf
from config.schema.providers import OpenAIProviderSettings, AzureOpenAIProviderSettings, Providers
from config.schema.misc import DatasetSettings
from config.schema.annotator import AnnotatorSettings, AnntatorEnum, ArgillaSettings, AnnotatorLlmSettings
from config.schema.predictor import PredictorSettings, PredictorEnum, PredictorLlmSettings
from config.schema.main import Settings

envvar_prefix="AUTOPROMPT",
toml_settings_files=['settings.toml', '.secrets.toml',]
yaml_settings_files=['config/config_default.yml',]

def load_dynaconf(envvar_prefix: str, settings_files: list[str]):
    return Dynaconf(
        envvar_prefix=envvar_prefix,
        settings_files=settings_files,
    )


default_loaded_settings = load_dynaconf(
    envvar_prefix="AUTOPROMPT",
    settings_files=toml_settings_files + yaml_settings_files,
)


# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.

def load_schema(load_settings=default_loaded_settings):
    if load_settings.openai.api_key:
        openai_provider = OpenAIProviderSettings(
            api_key=load_settings.openai.api_key,
            organization=load_settings.openai.get('organization', None)
        )
    elif load_settings.azure_openai.api_key and load_settings.azure_openai.endpoint and load_settings.azure_openai.api_version:
        openai_provider = AzureOpenAIProviderSettings(
            api_key=load_settings.azure_openai.api_key,
            endpoint=load_settings.azure_openai.endpoint,
            api_version=load_settings.azure_openai.api_version
        )

    annotator_method = AnntatorEnum(load_settings.annotator.method.lower())

    if annotator_method == AnntatorEnum.argilla:
        annotator_config = ArgillaSettings(**load_settings.annotator.config)
    elif annotator_method == AnntatorEnum.llm:
        annotator_config = AnnotatorLlmSettings(**load_settings.annotator.config)

    predictor_method = PredictorEnum(load_settings.predictor.method.lower())
    if predictor_method == PredictorEnum.llm:
        predictor_config = PredictorLlmSettings(

            **load_settings.predictor.config)

    settings = Settings(
        use_wandb=load_settings.use_wandb,
        providers=Providers(openai=openai_provider),
        dataset=DatasetSettings(**load_settings.dataset),
        annotator=AnnotatorSettings(
            method=annotator_method,
            config=annotator_config,
        ),
        predictor=PredictorSettings(**load_settings.predictor),
    )

    return settings


