"""
Ejemplo de uso del sistema RAG con RoBERTa-base para embeddings.
Este script demuestra cómo:
1. Procesar documentos con RoBERTa
2. Almacenar embeddings en MongoDB
3. Realizar búsquedas por similitud
4. Comparar con SentenceTransformers
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import logging
import time
from typing import List, Dict, Any

# Importar las clases del sistema
from src.document_processor_roberta import DocumentProcessorRoBERTa
from src.vector_store_roberta import MongoVectorStoreRoBERTa
from src.config import settings

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()


def demo_roberta_processing():
    """Demuestra el procesamiento de documentos con RoBERTa."""
    
    print("\n" + "="*60)
    print("DEMO: Procesamiento de Documentos con RoBERTa-base")
    print("="*60 + "\n")
    
    # 1. Inicializar el procesador con RoBERTa
    processor_roberta = DocumentProcessorRoBERTa(
        use_roberta=True,
        embedding_model="roberta-base",
        chunk_size=500,  # Chunks más pequeños para demostración
        chunk_overlap=100,
        device=settings.roberta_device,
        batch_size=settings.roberta_batch_size
    )
    
    # 2. Procesar un texto de ejemplo
    sample_text = """
    La inteligencia artificial está transformando el mundo. Los modelos de lenguaje
    como RoBERTa han demostrado ser muy efectivos para tareas de comprensión del
    lenguaje natural. RoBERTa es una versión optimizada de BERT que utiliza más
    datos de entrenamiento y técnicas mejoradas.
    
    En el campo del procesamiento del lenguaje natural, los embeddings son
    representaciones vectoriales densas que capturan el significado semántico
    del texto. Estos embeddings permiten realizar búsquedas semánticas eficientes
    y encontrar documentos relevantes basados en similitud.
    """
    
    print("Texto de ejemplo:")
    print("-" * 40)
    print(sample_text[:200] + "...")
    print("-" * 40 + "\n")
    
    # Procesar el texto
    chunks = processor_roberta.process_single_text(sample_text)
    
    print(f"✓ Texto procesado en {len(chunks)} chunks")
    print(f"✓ Dimensión de embeddings: {chunks[0]['embedding_dimension']}")
    print(f"✓ Modelo usado: RoBERTa-base\n")
    
    return chunks


def demo_vector_store():
    """Demuestra el almacenamiento y búsqueda con MongoDB."""
    
    print("\n" + "="*60)
    print("DEMO: Vector Store con MongoDB y RoBERTa")
    print("="*60 + "\n")
    
    # 1. Inicializar el vector store
    vector_store = MongoVectorStoreRoBERTa(
        mongodb_uri=settings.mongodb_uri,
        database_name=settings.mongodb_database,
        collection_name="roberta_test_collection",
        use_roberta=True,
        embedding_model="roberta-base",
        device=settings.roberta_device
    )
    
    # 2. Limpiar colección para la demo
    vector_store.clear_collection()
    print("✓ Colección limpiada\n")
    
    # 3. Crear y almacenar documentos de ejemplo
    documents = [
        {
            "title": "Introducción a Machine Learning",
            "content": """Machine Learning es una rama de la inteligencia artificial que permite
            a las computadoras aprender de los datos sin ser programadas explícitamente.
            Los algoritmos de ML pueden identificar patrones y hacer predicciones."""
        },
        {
            "title": "Deep Learning y Redes Neuronales",
            "content": """Deep Learning utiliza redes neuronales artificiales con múltiples capas
            para aprender representaciones jerárquicas de los datos. Es especialmente efectivo
            en visión por computadora y procesamiento del lenguaje natural."""
        },
        {
            "title": "Transformers en NLP",
            "content": """Los modelos Transformer como BERT, GPT y RoBERTa han revolucionado el
            procesamiento del lenguaje natural. Utilizan mecanismos de atención para capturar
            relaciones a largo plazo en el texto."""
        },
        {
            "title": "Embeddings y Representación Vectorial",
            "content": """Los embeddings son representaciones vectoriales densas que capturan
            el significado semántico del texto. Permiten medir similitudes y realizar
            búsquedas semánticas eficientes en grandes colecciones de documentos."""
        }
    ]
    
    # Procesar documentos
    processor = DocumentProcessorRoBERTa(use_roberta=True)
    all_chunks = []
    
    for doc in documents:
        chunks = processor.process_single_text(
            doc["content"],
            metadata={"title": doc["title"]}
        )
        all_chunks.extend(chunks)
    
    # Almacenar en MongoDB
    success = vector_store.add_documents(all_chunks)
    if success:
        print(f"✓ {len(all_chunks)} chunks almacenados en MongoDB\n")
    
    # 4. Realizar búsquedas de ejemplo
    queries = [
        "¿Qué son los transformers en inteligencia artificial?",
        "Explícame cómo funcionan los embeddings",
        "Diferencia entre machine learning y deep learning"
    ]
    
    print("Búsquedas de ejemplo:")
    print("-" * 40)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        
        # Búsqueda por similitud
        results = vector_store.similarity_search(
            query=query,
            top_k=2,
            similarity_threshold=0.5
        )
        
        print(f"Encontrados {len(results)} resultados relevantes:")
        for i, result in enumerate(results, 1):
            title = result.get("metadata", {}).get("title", "Sin título")
            similarity = result.get("similarity", 0)
            content_preview = result["content"][:100] + "..."
            print(f"  {i}. [{similarity:.3f}] {title}")
            print(f"     {content_preview}")
    
    # 5. Mostrar estadísticas
    stats = vector_store.get_collection_stats()
    print("\n" + "-" * 40)
    print("Estadísticas de la colección:")
    print(f"  • Total documentos: {stats['total_documents']}")
    print(f"  • IDs únicos: {stats['unique_doc_ids']}")
    print(f"  • Modelos de embedding: {stats['embedding_models']}")
    print(f"  • Tamaño promedio de chunk: {stats['average_chunk_size']:.0f} caracteres")


def compare_with_sentence_transformers():
    """Compara RoBERTa con SentenceTransformers."""
    
    print("\n" + "="*60)
    print("COMPARACIÓN: RoBERTa vs SentenceTransformers")
    print("="*60 + "\n")
    
    sample_text = """
    Los modelos de lenguaje pre-entrenados han revolucionado el procesamiento
    del lenguaje natural. Estos modelos aprenden representaciones ricas del
    lenguaje a partir de grandes cantidades de texto no etiquetado.
    """
    
    # Procesar con RoBERTa
    print("1. Procesando con RoBERTa-base...")
    start_time = time.time()
    processor_roberta = DocumentProcessorRoBERTa(
        use_roberta=True,
        embedding_model="roberta-base"
    )
    chunks_roberta = processor_roberta.process_single_text(sample_text)
    roberta_time = time.time() - start_time
    
    # Procesar con SentenceTransformers
    print("2. Procesando con SentenceTransformers...")
    start_time = time.time()
    processor_st = DocumentProcessorRoBERTa(
        use_roberta=False,
        embedding_model="all-MiniLM-L6-v2"
    )
    chunks_st = processor_st.process_single_text(sample_text)
    st_time = time.time() - start_time
    
    # Comparar resultados
    print("\n" + "-" * 40)
    print("Resultados de la comparación:")
    print("-" * 40)
    print(f"\nRoBERTa-base:")
    print(f"  • Dimensión de embeddings: {chunks_roberta[0]['embedding_dimension']}")
    print(f"  • Tiempo de procesamiento: {roberta_time:.3f}s")
    print(f"  • Modelo base: BERT mejorado")
    
    print(f"\nSentenceTransformers (all-MiniLM-L6-v2):")
    print(f"  • Dimensión de embeddings: {chunks_st[0]['embedding_dimension']}")
    print(f"  • Tiempo de procesamiento: {st_time:.3f}s")
    print(f"  • Modelo base: Optimizado para similitud")
    
    print("\n" + "-" * 40)
    print("Recomendaciones:")
    print("-" * 40)
    print("• RoBERTa-base: Mejor para tareas que requieren comprensión profunda del contexto")
    print("• SentenceTransformers: Más rápido y eficiente para búsqueda por similitud")
    print("• La elección depende de tu caso de uso específico y recursos disponibles")


def process_markdown_documents():
    """Procesa documentos Markdown reales si existen."""
    
    print("\n" + "="*60)
    print("PROCESAMIENTO DE DOCUMENTOS MARKDOWN")
    print("="*60 + "\n")
    
    # Buscar directorio de documentos
    docs_dir = Path("documents")
    
    if not docs_dir.exists():
        print("ℹ️  Creando directorio de documentos de ejemplo...")
        docs_dir.mkdir(exist_ok=True)
        
        # Crear documento de ejemplo
        example_doc = docs_dir / "ejemplo_roberta.md"
        example_content = """# RoBERTa: A Robustly Optimized BERT Pretraining Approach

## Introducción

RoBERTa es una reimplementación optimizada de BERT que modifica aspectos clave del entrenamiento:

1. **Entrenamiento más largo**: Más épocas con batches más grandes
2. **Eliminación de NSP**: Remueve la tarea de Next Sentence Prediction
3. **Máscaras dinámicas**: Cambia el patrón de máscara en cada época
4. **Datos más grandes**: Utiliza 10 veces más datos que BERT original

## Arquitectura

RoBERTa mantiene la misma arquitectura que BERT:
- **roberta-base**: 12 capas, 768 dimensiones hidden, 12 attention heads
- **roberta-large**: 24 capas, 1024 dimensiones hidden, 16 attention heads

## Aplicaciones en RAG

En sistemas RAG, RoBERTa puede usarse para:

### 1. Generación de Embeddings
Los embeddings de RoBERTa capturan información semántica rica que es útil para:
- Búsqueda por similitud
- Clustering de documentos
- Detección de duplicados

### 2. Comprensión de Consultas
RoBERTa puede entender mejor las intenciones detrás de las consultas de los usuarios.

### 3. Reranking
Usar RoBERTa para reordenar resultados basándose en relevancia semántica.

## Ventajas sobre BERT

- **Mejor rendimiento**: Supera a BERT en la mayoría de benchmarks
- **Más robusto**: Menos sensible a hiperparámetros
- **Transferencia efectiva**: Se adapta bien a tareas downstream

## Consideraciones de Implementación

- **Recursos computacionales**: Requiere GPU para procesamiento eficiente
- **Tamaño del modelo**: ~125M parámetros para roberta-base
- **Velocidad de inferencia**: Más lento que modelos distilados pero más preciso
"""
        
        with open(example_doc, 'w', encoding='utf-8') as f:
            f.write(example_content)
        
        print(f"✓ Creado documento de ejemplo: {example_doc}\n")
    
    # Procesar documentos
    processor = DocumentProcessorRoBERTa(
        use_roberta=True,
        embedding_model="roberta-base",
        chunk_size=500,
        chunk_overlap=100
    )
    
    # Procesar todos los archivos Markdown
    all_chunks = processor.process_documents(
        str(docs_dir),
        file_pattern="*.md",
        show_progress=True
    )
    
    if all_chunks:
        print(f"\n✓ Procesados {len(all_chunks)} chunks de documentos Markdown")
        
        # Almacenar en MongoDB
        vector_store = MongoVectorStoreRoBERTa(
            mongodb_uri=settings.mongodb_uri,
            database_name=settings.mongodb_database,
            collection_name="markdown_roberta_collection",
            use_roberta=True
        )
        
        vector_store.clear_collection()
        success = vector_store.add_documents(all_chunks)
        
        if success:
            print(f"✓ Chunks almacenados en MongoDB")
            
            # Realizar búsqueda de prueba
            test_query = "ventajas de RoBERTa sobre BERT"
            results = vector_store.similarity_search(test_query, top_k=3)
            
            print(f"\n📍 Búsqueda de prueba: '{test_query}'")
            print(f"   Encontrados {len(results)} resultados relevantes")
    else:
        print("⚠️  No se encontraron documentos Markdown para procesar")


def main():
    """Función principal que ejecuta todas las demos."""
    
    print("\n" + "="*70)
    print(" SISTEMA RAG CON RoBERTa-base EMBEDDINGS ")
    print("="*70)
    print("\nEste script demuestra el uso de RoBERTa para generar embeddings")
    print("en un sistema RAG (Retrieval-Augmented Generation).\n")
    
    try:
        # Verificar conexión a MongoDB
        test_store = MongoVectorStoreRoBERTa(
            mongodb_uri=settings.mongodb_uri,
            database_name=settings.mongodb_database,
            collection_name="test",
            use_roberta=True
        )
        
        if not test_store.health_check():
            print("⚠️  MongoDB no está disponible. Por favor, inicia MongoDB primero.")
            print("   Puedes iniciarlo con: brew services start mongodb-community")
            return
        
        print("✓ Conexión a MongoDB verificada\n")
        
        # Ejecutar demos
        input("Presiona Enter para comenzar con la demo de procesamiento...")
        chunks = demo_roberta_processing()
        
        input("\nPresiona Enter para continuar con la demo del vector store...")
        demo_vector_store()
        
        input("\nPresiona Enter para ver la comparación con SentenceTransformers...")
        compare_with_sentence_transformers()
        
        input("\nPresiona Enter para procesar documentos Markdown...")
        process_markdown_documents()
        
        print("\n" + "="*70)
        print(" DEMO COMPLETADA EXITOSAMENTE ")
        print("="*70)
        print("\n✅ Todas las funcionalidades han sido demostradas.")
        print("📚 Ahora puedes usar RoBERTa en tu sistema RAG.\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrumpida por el usuario.")
    except Exception as e:
        logger.error(f"Error en la demo: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
