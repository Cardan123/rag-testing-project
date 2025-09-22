"""
Script de prueba simple para RoBERTa sin necesidad de MongoDB.
Demuestra la generación de embeddings y búsqueda por similitud en memoria.
"""

import numpy as np
from typing import List, Dict, Any
import logging
from datetime import datetime

# Importar las clases
from src.roberta_embeddings import RoBERTaEmbeddings
from src.document_processor_roberta import DocumentProcessorRoBERTa

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_roberta_embeddings():
    """
    Prueba básica de generación de embeddings con RoBERTa.
    """
    print("\n" + "="*60)
    print("PRUEBA: Generación de Embeddings con RoBERTa")
    print("="*60 + "\n")
    
    # Inicializar RoBERTa
    print("Inicializando RoBERTa-base (puede tardar la primera vez)...")
    roberta = RoBERTaEmbeddings(
        model_name="roberta-base",
        device="cpu",  # Usar CPU para la prueba
        batch_size=4
    )
    print(f"✓ Modelo cargado. Dimensión de embeddings: {roberta.get_embedding_dimension()}\n")
    
    # Textos de ejemplo
    documents = [
        "The concrete cover for footings should be 7.5 cm according to specifications.",
        "Seismic design parameters include a zone factor of 0.45 for this region.",
        "Machine learning models like RoBERTa can understand technical documentation.",
        "The reduction coefficient varies between 6.0 and 8.0 depending on the axis.",
        "Python is a popular programming language for data science applications."
    ]
    
    print("Documentos de ejemplo:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc[:60]}...")
    
    # Generar embeddings
    print("\nGenerando embeddings para documentos...")
    doc_embeddings = roberta.encode_documents(
        documents,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    print(f"✓ Embeddings generados: shape {doc_embeddings.shape}")
    
    # Query de búsqueda
    query = "What are the seismic parameters and reduction factors?"
    print(f"\nQuery de búsqueda: '{query}'")
    
    # Generar embedding para query
    query_embedding = roberta.encode_queries(
        query,
        normalize_embeddings=True
    )
    
    # Calcular similitudes
    similarities = roberta.similarity(query_embedding.reshape(1, -1), doc_embeddings)
    similarities = similarities.flatten()
    
    # Ordenar por similitud
    ranked_indices = np.argsort(similarities)[::-1]
    
    print("\nResultados ordenados por relevancia:")
    print("-" * 40)
    for idx in ranked_indices:
        print(f"  [{similarities[idx]:.3f}] {documents[idx][:70]}...")
    
    return doc_embeddings, documents


def test_document_processing():
    """
    Prueba el procesamiento de documentos con la estructura correcta.
    """
    print("\n" + "="*60)
    print("PRUEBA: Procesamiento de Documentos")
    print("="*60 + "\n")
    
    # Inicializar procesador
    processor = DocumentProcessorRoBERTa(
        use_roberta=True,
        embedding_model="roberta-base",
        chunk_size=200,  # Chunks pequeños para la demo
        chunk_overlap=50
    )
    
    # Documento de ejemplo con estructura técnica
    technical_doc = """
    # Structural Design Specifications
    
    ## Concrete Requirements
    The minimum concrete strength for structural elements shall be:
    - Foundations: f'c = 280 kg/cm²
    - Columns and beams: f'c = 350 kg/cm²
    - Slabs: f'c = 280 kg/cm²
    
    ## Reinforcement Cover
    Minimum concrete cover for reinforcement:
    - Footings in contact with soil: 7.5 cm
    - Columns exposed to weather: 4.0 cm
    - Interior beams and slabs: 3.0 cm
    
    ## Seismic Design Parameters
    According to seismic code E.030:
    - Zone factor (Z): 0.45 (high seismicity)
    - Usage factor (U): 1.00 (residential)
    - Soil factor (S): 1.00 (rigid soil)
    - Reduction coefficient (R): 8.0 for X-axis, 6.0 for Y-axis
    """
    
    print("Procesando documento técnico...")
    chunks = processor.process_single_text(technical_doc)
    
    print(f"\n✓ Documento dividido en {len(chunks)} chunks")
    print("\nEstructura del primer chunk (simulando tu BD):")
    print("-" * 40)
    
    # Mostrar estructura sin embedding
    first_chunk = chunks[0].copy()
    embedding = first_chunk.pop("embedding", None)
    
    for key, value in first_chunk.items():
        if isinstance(value, str) and len(str(value)) > 50:
            print(f"  {key}: {str(value)[:50]}...")
        elif isinstance(value, datetime):
            print(f"  {key}: {value.isoformat()}")
        else:
            print(f"  {key}: {value}")
    
    print(f"  embedding: [array de {len(embedding)} dimensiones]")
    
    return chunks


def test_similarity_search_in_memory(chunks: List[Dict[str, Any]]):
    """
    Simula una búsqueda por similitud usando chunks en memoria.
    """
    print("\n" + "="*60)
    print("PRUEBA: Búsqueda por Similitud (sin MongoDB)")
    print("="*60 + "\n")
    
    # Inicializar RoBERTa para queries
    roberta = RoBERTaEmbeddings(model_name="roberta-base")
    
    # Queries de prueba
    queries = [
        "concrete cover requirements for footings",
        "seismic reduction coefficient",
        "minimum concrete strength for columns"
    ]
    
    print("Realizando búsquedas en los chunks procesados:")
    print("-" * 40)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        
        # Generar embedding para query
        query_embedding = roberta.encode_queries(query, normalize_embeddings=True)
        
        # Calcular similitudes con todos los chunks
        similarities = []
        for chunk in chunks:
            chunk_embedding = np.array(chunk["embedding"])
            # Normalizar chunk embedding
            chunk_embedding = chunk_embedding / np.linalg.norm(chunk_embedding)
            # Calcular similitud del coseno
            similarity = np.dot(query_embedding.flatten(), chunk_embedding)
            similarities.append(similarity)
        
        # Obtener top 2 resultados
        top_indices = np.argsort(similarities)[::-1][:2]
        
        print("Resultados más relevantes:")
        for rank, idx in enumerate(top_indices, 1):
            chunk_text = chunks[idx].get("chunk_text", "")[:100]
            print(f"  {rank}. [{similarities[idx]:.3f}] Chunk {chunks[idx]['chunk_index']}")
            print(f"     {chunk_text}...")


def compare_roberta_vs_sentence_transformers():
    """
    Compara RoBERTa con SentenceTransformers.
    """
    print("\n" + "="*60)
    print("COMPARACIÓN: RoBERTa vs SentenceTransformers")
    print("="*60 + "\n")
    
    sample_text = """
    The structural design must comply with seismic regulations.
    Concrete strength and reinforcement specifications are critical.
    """
    
    # Procesar con RoBERTa
    print("1. Procesando con RoBERTa-base...")
    processor_roberta = DocumentProcessorRoBERTa(
        use_roberta=True,
        embedding_model="roberta-base"
    )
    
    import time
    start = time.time()
    chunks_roberta = processor_roberta.process_single_text(sample_text)
    roberta_time = time.time() - start
    
    # Procesar con SentenceTransformers
    print("2. Procesando con SentenceTransformers...")
    processor_st = DocumentProcessorRoBERTa(
        use_roberta=False,
        embedding_model="all-MiniLM-L6-v2"
    )
    
    start = time.time()
    chunks_st = processor_st.process_single_text(sample_text)
    st_time = time.time() - start
    
    # Comparar
    print("\n" + "-" * 40)
    print("Resultados de la comparación:")
    print("-" * 40)
    
    print(f"\nRoBERTa-base:")
    print(f"  • Dimensión: {chunks_roberta[0]['dims']}")
    print(f"  • Tiempo: {roberta_time:.3f}s")
    print(f"  • Modelo: {chunks_roberta[0]['model']}")
    
    print(f"\nSentenceTransformers:")
    print(f"  • Dimensión: {chunks_st[0]['dims']}")
    print(f"  • Tiempo: {st_time:.3f}s")
    print(f"  • Modelo: {chunks_st[0]['model']}")
    
    print("\n💡 Recomendaciones:")
    print("  • RoBERTa: Mejor comprensión contextual, ideal para documentos técnicos")
    print("  • SentenceTransformers: Más rápido, optimizado para similitud")


def main():
    """
    Función principal que ejecuta todas las pruebas.
    """
    print("\n" + "="*70)
    print(" PRUEBA DE ROBERTA SIN MONGODB ")
    print("="*70)
    print("\nEste script demuestra las capacidades de RoBERTa")
    print("sin necesidad de una base de datos.\n")
    
    try:
        # 1. Prueba básica de embeddings
        print("📝 Paso 1: Probando generación de embeddings...")
        doc_embeddings, documents = test_roberta_embeddings()
        input("\nPresiona Enter para continuar...")
        
        # 2. Procesamiento de documentos
        print("\n📄 Paso 2: Procesando documentos con estructura correcta...")
        chunks = test_document_processing()
        input("\nPresiona Enter para continuar...")
        
        # 3. Búsqueda por similitud
        print("\n🔍 Paso 3: Simulando búsqueda por similitud...")
        test_similarity_search_in_memory(chunks)
        input("\nPresiona Enter para continuar...")
        
        # 4. Comparación de modelos
        print("\n📊 Paso 4: Comparando modelos...")
        compare_roberta_vs_sentence_transformers()
        
        print("\n" + "="*70)
        print(" PRUEBAS COMPLETADAS ")
        print("="*70)
        print("\n✅ RoBERTa está funcionando correctamente.")
        print("📚 Cuando MongoDB esté disponible, podrás usar el sistema completo.\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Pruebas interrumpidas por el usuario.")
    except Exception as e:
        logger.error(f"Error en las pruebas: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
