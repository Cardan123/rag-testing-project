import os
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    mongodb_uri: str = Field(default="mongodb://localhost:27017", env="MONGODB_URI")
    mongodb_database: str = Field(default="rag_testing", env="MONGODB_DATABASE")
    mongodb_collection: str = Field(default="documents", env="MONGODB_COLLECTION")

    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    gemini_model: str = Field(default="gemini-1.5-pro")

    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)

    top_k_retrieval: int = Field(default=5)
    similarity_threshold: float = Field(default=0.7)

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()