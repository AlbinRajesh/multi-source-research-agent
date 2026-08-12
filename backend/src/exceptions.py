"""Custom exceptions for the Research & Search Agent."""
from typing import Optional


class ResearchAgentError(Exception):
    """Base exception for all research agent errors."""

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        return f"{self.message}: {self.details}" if self.details else self.message


class DeepResearchError(ResearchAgentError):
    """Raised when a deep research operation fails globally."""
    pass


class ConfigurationError(ResearchAgentError):
    pass


class PlanningError(ResearchAgentError):
    pass


class SearchError(ResearchAgentError):
    pass


class RateLimitError(SearchError):
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60, service: str = "unknown"):
        super().__init__(message)
        self.retry_after = retry_after
        self.service = service


class ContentExtractionError(ResearchAgentError):
    def __init__(self, message: str, url: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.url = url
        self.status_code = status_code


class ClaimExtractionError(ResearchAgentError):
    """Errors during atomic claim extraction from documents."""
    pass


class VerificationError(ResearchAgentError):
    """Errors during claim groundedness verification."""
    pass


class SynthesisError(ResearchAgentError):
    pass


class ReportGenerationError(ResearchAgentError):
    pass


class CircuitOpenError(ResearchAgentError):
    def __init__(self, service: str, retry_after: float):
        super().__init__(f"Circuit breaker open for {service}")
        self.service = service
        self.retry_after = retry_after


class ValidationError(ResearchAgentError):
    pass


class LLMError(ResearchAgentError):
    def __init__(self, message: str, provider: str, model: Optional[str] = None, is_retryable: bool = True):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.is_retryable = is_retryable