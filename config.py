"""
Configuration module for Multi-Modal RAG Search Engine.

Manages all environment variables, model settings, and application constants.
"""

from pydantic_settings import BaseSettings
from loguru import logger


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    groq_api_key: str
    llm_model: str = "llama-3.3-70b-versatile"  # Latest, fast, recommended
    vision_model: str = "llama-2-90b-vision"  # For image description
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace embeddings

    # Vector Database
    chroma_path: str = "data/chroma_db"
    collection_name: str = "documents"

    # File Storage
    data_folder: str = "data"

    # Search Parameters
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5  # Number of documents to retrieve for context

    # UI Settings
    app_title: str = "Multi-Modal RAG Search Engine"
    app_icon: str = "🔍"

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
logger.info(f"Configuration loaded | LLM: {settings.llm_model}")
