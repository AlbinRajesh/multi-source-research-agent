"""LLM factory — single place that turns config into a LangChain chat model.

Every agent calls get_llm() rather than instantiating a provider directly,
so switching providers (ollama <-> hosted API) is a config change, not a
code change across 5 files.
"""
import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import config
from src.exceptions import ConfigurationError, LLMError

logger = logging.getLogger(__name__)


def get_llm(
    temperature: float = 0.3,
    model_override: Optional[str] = None,
    model_override_key: Optional[str] = None,
    provider_override: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> BaseChatModel:
    """Get an LLM instance based on config.

    Args:
        temperature: sampling temperature
        model_override: explicit model name, takes priority over everything
        model_override_key: name of a config field to read the model name
            from (e.g. "fast_model_name") — lets callers say "use the cheap
            model" without hardcoding a model name in the agent itself
        provider_override: explicit provider, overrides config.model_provider
        max_tokens: maximum tokens to generate in the completion
    """
    if model_override:
        model_name = model_override
    elif model_override_key:
        model_name = getattr(config, model_override_key, None)
        if not model_name:
            raise ConfigurationError(f"Unknown config field for model_override_key: {model_override_key}")
    else:
        model_name = config.model_name

    provider = provider_override or config.model_provider

    try:
        if provider == "ollama":
            logger.info(f"LLM: ollama/{model_name}")
            return ChatOllama(
                model=model_name,
                base_url=config.ollama_base_url,
                temperature=temperature,
                num_ctx=8192,
                num_predict=max_tokens if max_tokens else -1,
                # format="json" removed — caused this 3B model to abandon the
                # array-of-objects schema entirely (collapsing to a single string
                # or inconsistent wrapper keys) rather than producing valid JSON
                # in the requested shape. The prompt's "return ONLY a JSON array"
                # instruction + json_repair as a safety net proved more reliable
                # than Ollama's native JSON-mode for this model/schema combo.
            )

        elif provider == "openai":
            if not config.openai_api_key:
                raise ConfigurationError("OPENAI_API_KEY not set")
            logger.info(f"LLM: openai/{model_name}")
            return ChatOpenAI(
                model=model_name,
                base_url=f"{config.openai_base_url}/v1",
                api_key=config.openai_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        elif provider == "groq":
            if not config.groq_api_key:
                raise ConfigurationError("GROQ_API_KEY not set")
            logger.info(f"LLM: groq/{model_name}")
            return ChatOpenAI(
                model=model_name,
                base_url=config.groq_base_url,
                api_key=config.groq_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        elif provider == "nvidia":
            if not config.nvidia_api_key:
                raise ConfigurationError("NVIDIA_API_KEY not set")
            logger.info(f"LLM: nvidia/{model_name}")
            return ChatOpenAI(
                model=model_name,
                base_url=config.nvidia_base_url,
                api_key=config.nvidia_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        elif provider == "llamacpp":
            logger.info(f"LLM: llamacpp/{model_name}")
            return ChatOpenAI(
                model=model_name,
                base_url=f"{config.llamacpp_base_url}/v1",
                api_key="not-needed",
                temperature=temperature,
                max_tokens=max_tokens,
            )

        elif provider == "gemini":
            if not config.google_api_key:
                raise ConfigurationError("GOOGLE_API_KEY not set")
            logger.info(f"LLM: gemini/{model_name}")
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=config.google_api_key,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

        else:
            raise ConfigurationError(f"Unknown model_provider: {provider}")

    except ConfigurationError:
        raise
    except Exception as e:
        raise LLMError(f"Failed to initialize LLM: {e}", provider=provider, model=model_name)