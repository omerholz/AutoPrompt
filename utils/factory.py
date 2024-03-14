from config.schema.providers import (
    AnyProviderSettings,
    OpenAIProviderSettings,
    AzureOpenAIProviderSettings,
    GoogleAIProviderSettings,
    Providers,
)

from config.schema.llm import LlmSettings

from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


def llm(llm_settings: LlmSettings) -> LLMChain:
    if isinstance(llm_settings.provider, OpenAIProviderSettings):
        return ChatOpenAI(
            **llm_settings.provider.func_kwarg_model_dump()|llm_settings.model.func_kwarg_model_dump()
        )
    elif isinstance(llm_settings.provider, AzureOpenAIProviderSettings):
        return AzureChatOpenAI(
            **llm_settings.provider.func_kwarg_model_dump()|llm_settings.model.func_kwarg_model_dump()
        )
    elif isinstance(llm_settings.provider, GoogleAIProviderSettings):
        return ChatGoogleGenerativeAI(
            **llm_settings.provider.func_kwarg_model_dump()|llm_settings.model.func_kwarg_model_dump()
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_settings.provider}")




