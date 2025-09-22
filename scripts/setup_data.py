#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# Añadir el directorio padre al path para poder importar src
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Ahora importar las clases actualizadas con soporte RoBERTa
from src.document_processor_roberta import DocumentProcessorRoBERTa
from src.vector_store_roberta import MongoVectorStoreRoBERTa
from src.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_vector_database():
    logger.info("Setting up vector database with sample documents...")

    try:
        processor = DocumentProcessorRoBERTa(
            use_roberta=settings.use_roberta,
            embedding_model=settings.embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )

        vector_store = MongoVectorStoreRoBERTa(
            mongodb_uri=settings.mongodb_uri,
            database_name=settings.mongodb_database,
            collection_name=settings.mongodb_collection,
            use_roberta=settings.use_roberta,
            embedding_model=settings.embedding_model,
            device=settings.roberta_device,
            batch_size=settings.roberta_batch_size
        )

        # Verificar conexión a MongoDB
        try:
            stats = vector_store.get_collection_stats()
            logger.info(f"Conectado a MongoDB. Documentos actuales: {stats['total_documents']}")
        except Exception as e:
            logger.error(f"Cannot connect to MongoDB: {e}")
            return False

        documents_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'documents')

        if not os.path.exists(documents_dir):
            logger.error(f"Documents directory not found: {documents_dir}")
            return False

        logger.info("Processing documents and generating embeddings...")
        processed_chunks = processor.process_documents(documents_dir)

        if not processed_chunks:
            logger.warning("No documents were processed. Check your documents directory.")
            return False

        logger.info(f"Processed {len(processed_chunks)} document chunks")

        logger.info("Clearing existing collection...")
        vector_store.clear_collection()

        logger.info("Adding documents to MongoDB...")
        success = vector_store.add_documents(processed_chunks)

        if success:
            stats = vector_store.get_collection_stats()
            logger.info(f"Successfully added documents. Stats: {stats}")
            return True
        else:
            logger.error("Failed to add documents to MongoDB")
            return False

    except Exception as e:
        logger.error(f"Error setting up vector database: {e}")
        return False


def verify_setup():
    logger.info("Verifying setup...")

    try:
        vector_store = MongoVectorStoreRoBERTa(
            mongodb_uri=settings.mongodb_uri,
            database_name=settings.mongodb_database,
            collection_name=settings.mongodb_collection,
            use_roberta=settings.use_roberta,
            embedding_model=settings.embedding_model,
            device=settings.roberta_device,
            batch_size=settings.roberta_batch_size
        )

        stats = vector_store.get_collection_stats()
        logger.info(f"Database stats: {stats}")

        test_query = "What is machine learning?"
        results = vector_store.similarity_search(test_query, top_k=3)

        if results:
            logger.info(f"Test query successful! Found {len(results)} similar documents")
            for i, doc in enumerate(results[:2]):
                logger.info(f"  Document {i+1}: {doc['filename']} (similarity: {doc.get('similarity', 'N/A'):.3f})")
            return True
        else:
            logger.warning("Test query returned no results")
            return False

    except Exception as e:
        logger.error(f"Error during verification: {e}")
        return False


def main():
    logger.info("Starting RAG system setup...")

    logger.info("Checking environment variables...")
    required_vars = ['GEMINI_API_KEY']
    missing_vars = [var for var in required_vars if not getattr(settings, var.lower(), None)]

    if missing_vars:
        logger.error(f"Missing environment variables: {missing_vars}")
        logger.info("Please copy .env.example to .env and fill in the required values")
        return False

    if setup_vector_database():
        logger.info("Vector database setup completed successfully!")

        if verify_setup():
            logger.info("Setup verification passed!")
            logger.info("\nYou can now run the evaluation scripts:")
            logger.info("  python tests/test_ragas.py")
            logger.info("  python tests/test_deepeval.py")
            return True
        else:
            logger.warning("Setup verification failed, but database was created")
            return False
    else:
        logger.error("Setup failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)