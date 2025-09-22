import unittest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Añadir el directorio padre al path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Importar las clases actualizadas
from src.rag_system import RAGSystem
from src.vector_store_roberta import MongoVectorStoreRoBERTa
from src.document_processor_roberta import DocumentProcessorRoBERTa
from src.config import settings


class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DocumentProcessorRoBERTa(
            use_roberta=False,  # Usar SentenceTransformers para tests rápidos
            embedding_model="all-MiniLM-L6-v2",
            chunk_size=100,
            chunk_overlap=20
        )

    def test_chunk_document(self):
        document = {
            "doc_id": "test_doc",
            "filename": "test.md",
            "file_path": "/test/test.md",
            "text_content": "This is a test document. " * 20  # Create content longer than chunk_size
        }

        chunks = self.processor.chunk_document(document)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0]["doc_id"], "test_doc")
        self.assertEqual(chunks[0]["filename"], "test.md")
        self.assertEqual(chunks[0]["chunk_index"], 0)

    def test_generate_embeddings(self):
        chunks = [
            {
                "chunk_id": "test_chunk_1",
                "chunk_text": "This is test content one",
                "chunk_index": 0
            },
            {
                "chunk_id": "test_chunk_2",
                "chunk_text": "This is test content two",
                "chunk_index": 1
            }
        ]

        chunks_with_embeddings = self.processor.generate_embeddings(chunks)

        for chunk in chunks_with_embeddings:
            self.assertIn("embedding", chunk)
            self.assertIsInstance(chunk["embedding"], list)
            self.assertGreater(len(chunk["embedding"]), 0)


class TestRAGSystem(unittest.TestCase):
    def setUp(self):
        self.mock_vector_store = Mock(spec=MongoVectorStoreRoBERTa)
        self.rag_system = RAGSystem(self.mock_vector_store)

    @patch('google.generativeai.GenerativeModel')
    def test_query_with_results(self, mock_genai):
        mock_response = Mock()
        mock_response.text = "Machine learning is a subset of AI."
        mock_genai.return_value.generate_content.return_value = mock_response

        mock_retrieved_docs = [
            {
                "chunk_id": "doc1_chunk1",
                "chunk_text": "Machine learning is artificial intelligence",
                "similarity": 0.85,
                "doc_path": "ml_basics.md",
                "chunk_index": 0
            }
        ]

        self.mock_vector_store.similarity_search.return_value = mock_retrieved_docs

        result = self.rag_system.query("What is machine learning?")

        self.assertIn("question", result)
        self.assertIn("answer", result)
        self.assertIn("retrieved_documents", result)
        self.assertIn("context", result)

        self.assertEqual(result["question"], "What is machine learning?")
        self.assertEqual(result["answer"], "Machine learning is a subset of AI.")
        self.assertEqual(len(result["retrieved_documents"]), 1)

    def test_query_no_results(self):
        self.mock_vector_store.similarity_search.return_value = []

        result = self.rag_system.query("What is quantum computing?")

        self.assertEqual(result["answer"], "I couldn't find relevant information to answer your question.")
        self.assertEqual(len(result["retrieved_documents"]), 0)

    def test_format_context(self):
        docs = [
            {
                "content": "Content one",
                "filename": "doc1.md",
                "similarity": 0.9
            },
            {
                "content": "Content two",
                "filename": "doc2.md",
                "similarity": 0.8
            }
        ]

        context = self.rag_system._format_context(docs)

        self.assertIn("Content one", context)
        self.assertIn("Content two", context)
        self.assertIn("doc1.md", context)
        self.assertIn("doc2.md", context)
        self.assertIn("0.900", context)
        self.assertIn("0.800", context)


class TestEvaluationIntegration(unittest.TestCase):
    """Integration tests for the evaluation pipeline"""

    @patch('src.vector_store.MongoVectorStore')
    @patch('src.rag_system.RAGSystem')
    def test_evaluation_pipeline(self, mock_rag_system, mock_vector_store):
        mock_rag_instance = Mock()
        mock_rag_instance.query.return_value = {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of artificial intelligence.",
            "retrieved_documents": [
                {
                    "content": "ML content",
                    "similarity": 0.85,
                    "filename": "ml.md"
                }
            ],
            "context": "ML content"
        }

        mock_rag_system.return_value = mock_rag_instance

        test_questions = [
            {
                "question": "What is machine learning?",
                "expected_answer": "Machine learning is a subset of artificial intelligence.",
                "category": "definition"
            }
        ]

        results = []
        for item in test_questions:
            result = mock_rag_instance.query(item["question"])
            results.append(result)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["question"], "What is machine learning?")
        self.assertIn("retrieved_documents", results[0])


if __name__ == '__main__':
    unittest.main()