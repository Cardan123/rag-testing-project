import os
from typing import Optional
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Cargar .env explícitamente
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"✓ Archivo .env cargado desde: {env_path}")
else:
    print(f"⚠️ Archivo .env no encontrado en: {env_path}")
    # Intentar crear desde .env.example
    example_path = Path(__file__).parent.parent / '.env.example'
    if example_path.exists():
        import shutil
        shutil.copy(example_path, env_path)
        print(f"✓ Archivo .env creado desde .env.example")
        load_dotenv(dotenv_path=env_path, override=True)


class Settings(BaseSettings):
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    mongodb_uri: str = Field(default="mongodb://localhost:27017", env="MONGODB_URI")
    mongodb_database: str = Field(default="rag_testing", env="MONGODB_DATABASE")
    mongodb_collection: str = Field(default="documents", env="MONGODB_COLLECTION")

    # Modelo de embeddings - ahora soporta RoBERTa
    use_roberta: bool = Field(default=True, env="USE_ROBERTA")
    embedding_model: str = Field(default="roberta-base", env="EMBEDDING_MODEL")
    # Si use_roberta es False, se usa SentenceTransformers con este modelo:
    sentence_transformer_model: str = Field(default="all-MiniLM-L6-v2")
    
    # Configuración de RoBERTa
    roberta_device: Optional[str] = Field(default=None, env="ROBERTA_DEVICE")  # "cuda", "cpu", o None para auto
    roberta_batch_size: int = Field(default=8, env="ROBERTA_BATCH_SIZE")
    roberta_max_length: int = Field(default=512, env="ROBERTA_MAX_LENGTH")
    
    def __init__(self, **data):
        super().__init__(**data)
        # Limpiar roberta_device si tiene comentarios o está vacío
        if self.roberta_device:
            device_value = self.roberta_device.strip()
            # Si contiene un comentario o está vacío, establecer a None
            if not device_value or device_value.startswith('#') or device_value.startswith('//'):
                self.roberta_device = None
            elif device_value not in ['cuda', 'cpu', 'mps']:
                self.roberta_device = None
    
    gemini_model: str = Field(default="gemini-1.5-pro")

    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)

    top_k_retrieval: int = Field(default=5)
    similarity_threshold: float = Field(default=0.7)

    class Config:
        env_file = Path(__file__).parent.parent / '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False
        # Permitir campos extras si vienen del .env
        extra = "ignore"


settings = Settings()