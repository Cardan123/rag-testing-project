import os
import hashlib
from typing import List, Dict, Any
from pathlib import Path
import markdown
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np


class DocumentProcessor:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_markdown_file(self, file_path: str) -> Dict[str, Any]:
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
            "html_content": html
        }

    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = self.text_splitter.split_text(document["text_content"])

        chunked_docs = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document['doc_id']}_chunk_{i}"
            chunked_docs.append({
                "chunk_id": chunk_id,
                "doc_id": document["doc_id"],
                "filename": document["filename"],
                "file_path": document["file_path"],
                "chunk_index": i,
                "content": chunk,
                "metadata": {
                    "source": document["file_path"],
                    "chunk_size": len(chunk),
                    "total_chunks": len(chunks)
                }
            })

        return chunked_docs

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.tolist()

        return chunks

    def process_documents(self, document_dir: str) -> List[Dict[str, Any]]:
        all_chunks = []

        for md_file in Path(document_dir).glob("**/*.md"):
            document = self.load_markdown_file(str(md_file))
            chunks = self.chunk_document(document)
            chunks_with_embeddings = self.generate_embeddings(chunks)
            all_chunks.extend(chunks_with_embeddings)

        return all_chunks

    def _generate_doc_id(self, file_path: str, content: str) -> str:
        content_hash = hashlib.md5(content.encode()).hexdigest()
        filename = Path(file_path).stem
        return f"{filename}_{content_hash[:8]}"