"""LLM Provider factory for supporting multiple LLM backends."""

import os
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Sentinel value to detect unset parameters
_UNSET = object()


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def get_llm(self):
        """Return the LLM instance."""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        max_tokens: int = 1000,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.kwargs = kwargs
        self._llm = None
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    def get_llm(self):
        if self._llm is None:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                raise ImportError(
                    "Provider 'openai' requires 'langchain-openai'. "
                    "Install with: pip install langchain-openai"
                )
            self._llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.api_key,
                base_url=self.base_url,
                **self.kwargs
            )
        return self._llm


class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI LLM provider."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        max_tokens: int = 1000,
        api_version: str = "2024-02-01",
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        **kwargs
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_version = api_version
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.kwargs = kwargs
        self._llm = None
    
    @property
    def provider_name(self) -> str:
        return "azure"
    
    def get_llm(self):
        if self._llm is None:
            try:
                from langchain_openai import AzureChatOpenAI
            except ImportError:
                raise ImportError(
                    "Provider 'azure' requires 'langchain-openai'. "
                    "Install with: pip install langchain-openai"
                )
            self._llm = AzureChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                azure_deployment=self.azure_deployment,
                api_version=self.api_version,
                **self.kwargs
            )
        return self._llm


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider."""
    
    def __init__(
        self,
        model_name: str = "claude-3-haiku-20240307",
        temperature: float = 0,
        max_tokens: int = 1000,
        api_key: Optional[str] = None,
        **kwargs
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.kwargs = kwargs
        self._llm = None
    
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
    def get_llm(self):
        if self._llm is None:
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError:
                raise ImportError(
                    "Provider 'anthropic' requires 'langchain-anthropic'. "
                    "Install with: pip install langchain-anthropic"
                )
            self._llm = ChatAnthropic(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                anthropic_api_key=self.api_key,
                **self.kwargs
            )
        return self._llm


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider."""
    
    def __init__(
        self,
        model_name: str = "llama2",
        temperature: float = 0,
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.base_url = base_url
        self.kwargs = kwargs
        self._llm = None
    
    @property
    def provider_name(self) -> str:
        return "ollama"
    
    def get_llm(self):
        if self._llm is None:
            try:
                from langchain_ollama import ChatOllama
            except ImportError:
                raise ImportError(
                    "Provider 'ollama' requires 'langchain-ollama'. "
                    "Install with: pip install langchain-ollama"
                )
            self._llm = ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
                base_url=self.base_url,
                **self.kwargs
            )
        return self._llm


class GoogleAIProvider(BaseLLMProvider):
    """Google AI (Gemini) LLM provider."""
    
    def __init__(
        self,
        model_name: str = "gemini-pro",
        temperature: float = 0,
        max_output_tokens: int = 1000,
        api_key: Optional[str] = None,
        **kwargs
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.kwargs = kwargs
        self._llm = None
    
    @property
    def provider_name(self) -> str:
        return "google"
    
    def get_llm(self):
        if self._llm is None:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError:
                raise ImportError(
                    "Provider 'google' requires 'langchain-google-genai'. "
                    "Install with: pip install langchain-google-genai"
                )
            # Strip 'google/' prefix if present since Gemini API expects short model ID
            model_name = self.model_name.replace("google/", "")
            self._llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=self.temperature,
                convert_system_message_to_human=True,
                max_output_tokens=self.max_output_tokens,
                google_api_key=self.api_key,
                **self.kwargs
            )
        return self._llm


class AWSBedrockProvider(BaseLLMProvider):
    """AWS Bedrock LLM provider."""
    
    def __init__(
        self,
        model_name: str = "anthropic.claude-3-haiku-20240307-v1:0",
        temperature: float = 0,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None,
        **kwargs
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.kwargs = kwargs
        self._llm = None
    
    @property
    def provider_name(self) -> str:
        return "aws_bedrock"
    
    def get_llm(self):
        if self._llm is None:
            try:
                from langchain_aws import ChatBedrock
            except ImportError:
                raise ImportError(
                    "Provider 'aws_bedrock' requires 'langchain-aws'. "
                    "Install with: pip install langchain-aws"
                )
            self._llm = ChatBedrock(
                model_id=self.model_name,
                temperature=self.temperature,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region_name,
                **self.kwargs
            )
        return self._llm


# Provider registry
PROVIDERS: Dict[str, type] = {
    "openai": OpenAIProvider,
    "azure": AzureOpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
    "google": GoogleAIProvider,
    "aws_bedrock": AWSBedrockProvider,
}


def get_llm_provider(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature=_UNSET,
    max_tokens: int = 1000,
    **kwargs
) -> BaseLLMProvider:
    """
    Factory function to get an LLM provider.
    
    Args:
        provider: Provider name (openai, azure, anthropic, ollama, google, aws_bedrock).
                  If None, reads from LLM_PROVIDER env var or defaults to "openai".
        model_name: Model name (provider-specific).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        **kwargs: Additional provider-specific arguments.
    
    Returns:
        BaseLLMProvider instance.
    
    Raises:
        ValueError: If provider is not supported.
    """
    # Get provider from env if not specified
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    # Get default model from env if not specified
    if model_name is None:
        model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    # Get temperature from env if not specified (use _UNSET sentinel)
    if temperature is _UNSET and os.getenv("LLM_TEMPERATURE"):
        temperature = float(os.getenv("LLM_TEMPERATURE"))
    elif temperature is _UNSET:
        temperature = 0
    
    provider_class = PROVIDERS.get(provider.lower())
    if provider_class is None:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Supported providers: {list(PROVIDERS.keys())}"
        )
    
    logger.info(f"Initializing LLM provider: {provider} with model: {model_name}")
    
    return provider_class(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )


def create_llm(provider: Optional[str] = None, model_name: Optional[str] = None, **kwargs):
    """
    Convenience function to directly create an LLM instance.
    
    Args:
        provider: Provider name (defaults to LLM_PROVIDER env var or "openai").
        model_name: Model name (defaults to LLM_MODEL env var or "gpt-4o-mini").
        **kwargs: Additional arguments passed to get_llm_provider.
    
    Returns:
        LLM instance (ChatModel).
    """
    llm_provider = get_llm_provider(
        provider=provider,
        model_name=model_name,
        **kwargs
    )
    return llm_provider.get_llm()
