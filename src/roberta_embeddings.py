import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
from typing import List, Union, Optional
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class RoBERTaEmbeddings:
    """
    Una clase para generar embeddings usando RoBERTa-base.
    Optimizada para manejar textos largos y batches eficientemente.
    """
    
    def __init__(
        self, 
        model_name: str = "roberta-base",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 8
    ):
        """
        Inicializa el modelo RoBERTa para generar embeddings.
        
        Args:
            model_name: Nombre del modelo de Hugging Face (default: "roberta-base")
            device: Dispositivo para ejecutar el modelo ("cuda", "cpu", o None para auto-detección)
            max_length: Longitud máxima de tokens por texto
            batch_size: Tamaño del batch para procesamiento
        """
        # Configurar dispositivo
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Usando dispositivo: {self.device}")
        
        # Cargar tokenizador y modelo
        logger.info(f"Cargando modelo {model_name}...")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()  # Poner en modo evaluación
        
        self.max_length = max_length
        self.batch_size = batch_size
        self.embedding_dimension = self.model.config.hidden_size
        
        logger.info(f"Modelo cargado. Dimensión de embeddings: {self.embedding_dimension}")
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        convert_to_tensor: bool = False,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Genera embeddings para los textos proporcionados.
        
        Args:
            texts: Un texto o lista de textos
            convert_to_tensor: Si devolver tensores de PyTorch en lugar de numpy arrays
            show_progress_bar: Si mostrar barra de progreso
            normalize_embeddings: Si normalizar los embeddings (útil para cosine similarity)
        
        Returns:
            Embeddings como numpy array o tensor de PyTorch
        """
        # Convertir texto único a lista
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        all_embeddings = []
        
        # Procesar en batches
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        iterator = range(0, len(texts), self.batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Generando embeddings", total=num_batches)
        
        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + self.batch_size]
                
                # Tokenizar batch
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # Mover a dispositivo
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                
                # Generar embeddings
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Usar mean pooling sobre los tokens (excluyendo padding)
                embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
                
                # Normalizar si es necesario
                if normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu())
        
        # Concatenar todos los embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Si era un solo texto, devolver un solo embedding
        if single_text:
            all_embeddings = all_embeddings[0]
        
        # Convertir a numpy si no se requiere tensor
        if not convert_to_tensor:
            all_embeddings = all_embeddings.numpy()
        
        return all_embeddings
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Aplica mean pooling a los embeddings de tokens.
        
        Args:
            token_embeddings: Embeddings de todos los tokens
            attention_mask: Máscara de atención para ignorar padding
        
        Returns:
            Embeddings pooled
        """
        # Expandir máscara de atención
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sumar embeddings ponderados por la máscara
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Calcular el número de tokens no-padding
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        
        # Calcular promedio
        return sum_embeddings / sum_mask
    
    def encode_queries(self, queries: Union[str, List[str]], **kwargs) -> Union[np.ndarray, torch.Tensor]:
        """
        Método específico para codificar queries de búsqueda.
        Puede añadir preprocesamiento específico si es necesario.
        """
        if isinstance(queries, str):
            queries = [queries]
        
        # Podemos añadir un prefijo especial para queries si es necesario
        queries_with_prefix = [f"Query: {q}" for q in queries]
        
        return self.encode(queries_with_prefix, **kwargs)
    
    def encode_documents(self, documents: Union[str, List[str]], **kwargs) -> Union[np.ndarray, torch.Tensor]:
        """
        Método específico para codificar documentos.
        Puede añadir preprocesamiento específico si es necesario.
        """
        if isinstance(documents, str):
            documents = [documents]
        
        # Podemos añadir un prefijo especial para documentos si es necesario
        docs_with_prefix = [f"Document: {d}" for d in documents]
        
        return self.encode(docs_with_prefix, **kwargs)
    
    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Calcula la similitud del coseno entre dos conjuntos de embeddings.
        
        Args:
            embeddings1: Primer conjunto de embeddings
            embeddings2: Segundo conjunto de embeddings
        
        Returns:
            Matriz de similitudes
        """
        # Normalizar embeddings si no están normalizados
        embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=-1, keepdims=True)
        embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=-1, keepdims=True)
        
        # Calcular producto punto (similitud del coseno para vectores normalizados)
        return np.dot(embeddings1_norm, embeddings2_norm.T)
    
    def get_embedding_dimension(self) -> int:
        """Devuelve la dimensión de los embeddings."""
        return self.embedding_dimension
