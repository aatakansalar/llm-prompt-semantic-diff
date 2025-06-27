from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for the given text."""
        pass

class OpenAIProvider(EmbeddingProvider):
    """OpenAI API-based embedding provider."""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        try:
            import openai
            self.client = openai.OpenAI()
            self.model = model
        except ImportError as e:
            raise ImportError(
                "openai is required for OpenAI embeddings. "
                "Install with: pip install openai"
            ) from e
    
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings using OpenAI API."""
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

class SentenceTransformersProvider(EmbeddingProvider):
    """Local sentence-transformers-based embedding provider."""
    
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model)
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            ) from e
    
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings using sentence-transformers."""
        embeddings = self.model.encode([text], convert_to_tensor=False)[0]
        return embeddings.tolist()

def get_embeddings(
    text: str,
    provider: str = "sentence-transformers",
    model: Optional[str] = None,
) -> List[float]:
    """Get embeddings for text using the specified provider."""
    if provider == "openai":
        provider_instance = OpenAIProvider(model=model or "text-embedding-3-small")
    else:  # default to sentence-transformers
        provider_instance = SentenceTransformersProvider(model=model or "all-MiniLM-L6-v2")
    
    return provider_instance.get_embeddings(text) 