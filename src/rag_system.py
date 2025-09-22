from typing import List, Dict, Any, Optional, Union
import google.generativeai as genai
from .vector_store_roberta import MongoVectorStoreRoBERTa
from .config import settings
import logging

logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(self, vector_store: Union[MongoVectorStoreRoBERTa, Any]):
        self.vector_store = vector_store
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)

    def retrieve_documents(self, query: str, top_k: int = None, similarity_threshold: float = None) -> List[Dict[str, Any]]:
        top_k = top_k or settings.top_k_retrieval
        similarity_threshold = similarity_threshold or settings.similarity_threshold

        return self.vector_store.similarity_search(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

    def generate_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        context = self._format_context(retrieved_docs)

        prompt = f"""You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer the question. If the context doesn't contain
enough information to answer the question, say so clearly.

Context:
{context}

Question: {query}

Answer:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while generating a response."

    def query(self, question: str, top_k: int = None, similarity_threshold: float = None) -> Dict[str, Any]:
        retrieved_docs = self.retrieve_documents(question, top_k, similarity_threshold)

        if not retrieved_docs:
            return {
                "question": question,
                "answer": "I couldn't find relevant information to answer your question.",
                "retrieved_documents": [],
                "context": ""
            }

        answer = self.generate_response(question, retrieved_docs)
        context = self._format_context(retrieved_docs)

        return {
            "question": question,
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            "context": context
        }

    def _format_context(self, docs: List[Dict[str, Any]]) -> str:
        if not docs:
            return ""

        context_parts = []
        for i, doc in enumerate(docs, 1):
            # Usar doc_path o filename, lo que esté disponible
            source = doc.get('doc_path', doc.get('filename', 'Unknown source'))
            # Usar chunk_text o content, lo que esté disponible
            content = doc.get('chunk_text', doc.get('content', ''))
            similarity = doc.get('similarity', 0)

            context_parts.append(
                f"Document {i} (Source: {source}, Similarity: {similarity:.3f}):\n{content}\n"
            )

        return "\n".join(context_parts)

    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        results = []
        for question in questions:
            result = self.query(question)
            results.append(result)
        return results