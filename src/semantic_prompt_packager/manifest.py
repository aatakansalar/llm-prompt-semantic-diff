from typing import List, Optional
from pydantic import BaseModel, Field

class PromptManifest(BaseModel):
    """Model for a versioned prompt manifest."""
    
    content: str = Field(..., description="The prompt content")
    version: str = Field(..., description="Semantic version of the prompt")
    embeddings: List[float] = Field(..., description="Embedding vector for semantic comparison")
    description: Optional[str] = Field(None, description="Optional description of the prompt")
    tags: List[str] = Field(default_factory=list, description="Optional tags for categorization")
    model: str = Field(default="gpt-4", description="Target model identifier (e.g. gpt-4)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "Given a text passage, identify the main themes...",
                "version": "0.1.0",
                "embeddings": [0.1, -0.2, 0.3],
                "description": "Theme identification prompt",
                "tags": ["analysis", "themes"],
                "model": "gpt-4"
            }
        } 