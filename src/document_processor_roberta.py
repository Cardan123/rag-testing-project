import os
import hashlib
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import markdown
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from .roberta_embeddings import RoBERTaEmbeddings

logger = logging.getLogger(__name__)


class DocumentProcessorRoBERTa:
    """
    Procesador de documentos mejorado que puede usar RoBERTa o SentenceTransformers
    para generar embeddings.
    """
    
    def __init__(
        self, 
        embedding_model: Optional[str] = None,
        use_roberta: bool = True,
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        device: Optional[str] = None,
        batch_size: int = 8
    ):
        """
        Inicializa el procesador de documentos.
        
        Args:
            embedding_model: Nombre del modelo de embeddings
            use_roberta: Si usar RoBERTa (True) o SentenceTransformers (False)
            chunk_size: Tamaño de los chunks de texto
            chunk_overlap: Solapamiento entre chunks
            device: Dispositivo para el modelo ("cuda", "cpu", o None para auto)
            batch_size: Tamaño del batch para procesamiento
        """
        self.use_roberta = use_roberta
        
        if use_roberta:
            model_name = embedding_model or "roberta-base"
            logger.info(f"Inicializando RoBERTa con modelo: {model_name}")
            self.embedding_model = RoBERTaEmbeddings(
                model_name=model_name,
                device=device,
                batch_size=batch_size
            )
        else:
            model_name = embedding_model or "all-MiniLM-L6-v2"
            logger.info(f"Inicializando SentenceTransformers con modelo: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.info(f"DocumentProcessor inicializado con chunks de {chunk_size} caracteres")

    def load_markdown_file(self, file_path: str) -> Dict[str, Any]:
        """
        Carga un archivo Markdown y extrae su contenido.
        
        Args:
            file_path: Ruta al archivo Markdown
            
        Returns:
            Diccionario con información del documento
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        html = markdown.markdown(content, extensions=['meta', 'toc'])
        soup = BeautifulSoup(html, 'html.parser')
        text_content = soup.get_text()

        doc_id = self._generate_doc_id(file_path, content)

        return {
            "doc_id": doc_id,
            "file_path": file_path,
            "filename": Path(file_path).name,
            "content": content,
            "text_content": text_content,
            "html_content": html,
            "embedding_model": "roberta-base" if self.use_roberta else "sentence-transformers"
        }

    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Divide un documento en chunks más pequeños.
        Adaptado para la estructura existente en MongoDB.
        
        Args:
            document: Diccionario con información del documento
            
        Returns:
            Lista de chunks con metadata
        """
        chunks = self.text_splitter.split_text(document["text_content"])

        chunked_docs = []
        for i, chunk in enumerate(chunks):
            chunked_docs.append({
                # Campos principales para la estructura existente
                "doc_path": document["file_path"],
                "model": "roberta-base" if self.use_roberta else "sentence-transformers",
                "chunk_index": i,
                "chunk_method": "recursive",  # O el método que prefieras
                "chunk_text": chunk,
                "dims": self.embedding_model.get_embedding_dimension() if self.use_roberta else self.embedding_model.get_sentence_embedding_dimension(),
                
                # Campos adicionales útiles
                "doc_id": document["doc_id"],
                "filename": document["filename"],
                "metadata": {
                    "source": document["file_path"],
                    "chunk_size": len(chunk),
                    "total_chunks": len(chunks),
                    "embedding_model": "roberta-base" if self.use_roberta else "all-MiniLM-L6-v2"
                }
            })

        logger.info(f"Documento dividido en {len(chunks)} chunks")
        return chunked_docs

    def generate_embeddings(
        self, 
        chunks: List[Dict[str, Any]], 
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Genera embeddings para una lista de chunks.
        Adaptado para usar el campo 'chunk_text' de la estructura existente.
        
        Args:
            chunks: Lista de chunks de documentos
            show_progress: Si mostrar barra de progreso
            
        Returns:
            Lista de chunks con embeddings añadidos
        """
        # Usar 'chunk_text' si existe, sino buscar 'content' (retrocompatibilidad)
        texts = [chunk.get("chunk_text", chunk.get("content", "")) for chunk in chunks]
        
        logger.info(f"Generando embeddings para {len(texts)} chunks...")
        
        if self.use_roberta:
            # Usar RoBERTa para embeddings de documentos
            embeddings = self.embedding_model.encode_documents(
                texts,
                convert_to_tensor=False,
                show_progress_bar=show_progress,
                normalize_embeddings=True
            )
        else:
            # Usar SentenceTransformers
            embeddings = self.embedding_model.encode(
                texts, 
                convert_to_tensor=False,
                show_progress_bar=show_progress
            )

        # Añadir embeddings a los chunks y timestamps
        from datetime import datetime
        current_time = datetime.utcnow()
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.tolist() if hasattr(embedding, 'tolist') else embedding.tolist()
            chunk["dims"] = len(chunk["embedding"])  # Usar 'dims' como en tu estructura
            chunk["created_at"] = current_time
            chunk["updated_at"] = current_time
            
            # Asegurar que el modelo esté correctamente especificado
            if "model" not in chunk:
                chunk["model"] = "roberta-base" if self.use_roberta else "all-MiniLM-L6-v2"

        logger.info(f"Embeddings generados. Dimensión: {chunks[0]['dims']}")
        return chunks

    def process_documents(
        self, 
        document_dir: str, 
        file_pattern: str = "**/*.md",
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Procesa todos los documentos en un directorio.
        
        Args:
            document_dir: Directorio con documentos
            file_pattern: Patrón para buscar archivos
            show_progress: Si mostrar progreso
            
        Returns:
            Lista de todos los chunks procesados con embeddings
        """
        all_chunks = []
        document_path = Path(document_dir)
        
        # Buscar todos los archivos que coincidan con el patrón
        files = list(document_path.glob(file_pattern))
        logger.info(f"Encontrados {len(files)} archivos para procesar")

        for md_file in files:
            logger.info(f"Procesando: {md_file}")
            
            try:
                # Cargar documento
                document = self.load_markdown_file(str(md_file))
                
                # Dividir en chunks
                chunks = self.chunk_document(document)
                
                # Generar embeddings
                chunks_with_embeddings = self.generate_embeddings(chunks, show_progress=show_progress)
                
                all_chunks.extend(chunks_with_embeddings)
                
            except Exception as e:
                logger.error(f"Error procesando {md_file}: {e}")
                continue

        logger.info(f"Procesamiento completo. Total de chunks: {len(all_chunks)}")
        return all_chunks

    def process_single_text(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Procesa un texto individual sin necesidad de archivo.
        
        Args:
            text: Texto a procesar
            metadata: Metadata adicional opcional
            
        Returns:
            Lista de chunks procesados con embeddings
        """
        # Crear documento virtual
        doc_id = self._generate_doc_id("virtual", text)
        document = {
            "doc_id": doc_id,
            "file_path": "virtual",
            "filename": "virtual_document",
            "content": text,
            "text_content": text,
            "html_content": "",
            "embedding_model": "roberta-base" if self.use_roberta else "sentence-transformers"
        }
        
        if metadata:
            document.update(metadata)
        
        # Procesar como documento normal
        chunks = self.chunk_document(document)
        chunks_with_embeddings = self.generate_embeddings(chunks, show_progress=False)
        
        return chunks_with_embeddings

    def _generate_doc_id(self, file_path: str, content: str) -> str:
        """
        Genera un ID único para un documento.
        
        Args:
            file_path: Ruta del archivo
            content: Contenido del documento
            
        Returns:
            ID único del documento
        """
        content_hash = hashlib.md5(content.encode()).hexdigest()
        filename = Path(file_path).stem if file_path != "virtual" else "virtual"
        return f"{filename}_{content_hash[:8]}"
    
    def get_embedding_dimension(self) -> int:
        """Obtiene la dimensión de los embeddings."""
        if self.use_roberta:
            return self.embedding_model.get_embedding_dimension()
        else:
            # Para SentenceTransformers
            return self.embedding_model.get_sentence_embedding_dimension()
