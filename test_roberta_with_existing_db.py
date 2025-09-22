"""
Script de prueba para usar RoBERTa con la estructura de documentos existente en MongoDB.
Este script está adaptado para trabajar con documentos que tienen la estructura:
- doc_path, model, chunk_index, chunk_method, chunk_text, dims, embedding
"""

import os
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any
from datetime import datetime

# Importar las clases adaptadas
from src.vector_store_roberta import MongoVectorStoreRoBERTa
from src.document_processor_roberta import DocumentProcessorRoBERTa
from src.config import settings

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()


def test_similarity_search_existing_db():
    """
    Prueba la búsqueda por similitud con documentos existentes en la base de datos.
    """
    print("\n" + "="*60)
    print("PRUEBA: Búsqueda en Base de Datos Existente")
    print("="*60 + "\n")
    
    # Inicializar vector store con RoBERTa
    vector_store = MongoVectorStoreRoBERTa(
        mongodb_uri=settings.mongodb_uri,
        database_name=settings.mongodb_database,
        collection_name=settings.mongodb_collection,
        use_roberta=True,
        embedding_model="roberta-base",
        device=settings.roberta_device
    )
    
    # Verificar estadísticas de la colección
    stats = vector_store.get_collection_stats()
    print("Estadísticas de la colección existente:")
    print(f"  • Total de documentos: {stats['total_documents']}")
    print(f"  • Paths únicos: {stats['unique_doc_paths']}")
    print(f"  • Modelos usados: {stats['models_used']}")
    print(f"  • Métodos de chunking: {stats['chunk_methods']}")
    print(f"  • Dimensiones de embeddings: {stats['embedding_dimensions']}\n")
    
    # Realizar búsquedas de ejemplo
    queries = [
        "concrete cover specifications",
        "seismic design parameters",
        "live loads for basements",
        "reduction coefficient for structural design"
    ]
    
    print("Realizando búsquedas de ejemplo:")
    print("-" * 40)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        
        # Búsqueda por similitud con campos específicos
        results = vector_store.similarity_search(
            query=query,
            top_k=3,
            similarity_threshold=0.3,  # Umbral más bajo para documentos técnicos
            return_fields=["doc_path", "chunk_text", "chunk_index", "model", "similarity"]
        )
        
        if results:
            print(f"Encontrados {len(results)} resultados:")
            for i, result in enumerate(results, 1):
                similarity = result.get("similarity", 0)
                chunk_text_preview = result.get("chunk_text", "")[:150] + "..."
                doc_path = result.get("doc_path", "Unknown").split("/")[-1]  # Solo el nombre del archivo
                chunk_idx = result.get("chunk_index", "?")
                
                print(f"  {i}. [{similarity:.3f}] {doc_path} (chunk {chunk_idx})")
                print(f"     {chunk_text_preview}")
        else:
            print("  No se encontraron resultados.")


def add_new_document_with_structure():
    """
    Añade un nuevo documento usando la estructura correcta.
    """
    print("\n" + "="*60)
    print("AÑADIR: Nuevo Documento con Estructura Correcta")
    print("="*60 + "\n")
    
    # Inicializar procesador con RoBERTa
    processor = DocumentProcessorRoBERTa(
        use_roberta=True,
        embedding_model="roberta-base",
        chunk_size=500,
        chunk_overlap=100
    )
    
    # Documento de ejemplo sobre RoBERTa
    sample_doc = """
    # RoBERTa Technical Specifications
    
    ## Model Architecture
    RoBERTa uses the same architecture as BERT but with improved training methodology:
    - 12 layers (base) or 24 layers (large)
    - 768 hidden dimensions (base) or 1024 (large)
    - 12 attention heads (base) or 16 (large)
    
    ## Training Improvements
    1. Dynamic masking: Pattern changes each epoch
    2. No NSP (Next Sentence Prediction) task
    3. Larger batches: 8K sequences
    4. More training data: 160GB of text
    
    ## Performance Metrics
    - GLUE Score: 88.5 (base), 90.2 (large)
    - SQuAD 2.0 F1: 83.6 (base), 89.4 (large)
    - RACE Accuracy: 83.2% (base), 90.9% (large)
    """
    
    # Procesar el documento
    chunks = processor.process_single_text(
        sample_doc,
        metadata={"source": "roberta_technical_doc"}
    )
    
    # Asegurar que los chunks tengan la estructura correcta
    for chunk in chunks:
        # Verificar campos requeridos
        if "doc_path" not in chunk:
            chunk["doc_path"] = "example_docs/roberta_specs.md"
        if "chunk_method" not in chunk:
            chunk["chunk_method"] = "recursive"
    
    print(f"Documento procesado en {len(chunks)} chunks")
    print(f"Estructura del primer chunk:")
    
    # Mostrar estructura sin el embedding
    first_chunk = chunks[0].copy()
    first_chunk.pop("embedding", None)  # Remover embedding para visualización
    for key, value in first_chunk.items():
        if isinstance(value, str) and len(value) > 50:
            print(f"  • {key}: {value[:50]}...")
        else:
            print(f"  • {key}: {value}")
    
    # Inicializar vector store
    vector_store = MongoVectorStoreRoBERTa(
        mongodb_uri=settings.mongodb_uri,
        database_name=settings.mongodb_database,
        collection_name=settings.mongodb_collection,
        use_roberta=True
    )
    
    # Añadir documentos
    success = vector_store.add_documents(chunks)
    if success:
        print(f"\n✓ {len(chunks)} chunks añadidos exitosamente a MongoDB")
        
        # Verificar búsqueda con el nuevo documento
        test_query = "RoBERTa training improvements"
        results = vector_store.similarity_search(test_query, top_k=2)
        
        if results:
            print(f"\nPrueba de búsqueda: '{test_query}'")
            print(f"Encontrado en el nuevo documento: {len(results)} resultados")
    else:
        print("\n✗ Error al añadir documentos")


def compare_models_on_existing_data():
    """
    Compara los resultados de búsqueda entre diferentes modelos de embeddings.
    """
    print("\n" + "="*60)
    print("COMPARACIÓN: RoBERTa vs Modelo Existente")
    print("="*60 + "\n")
    
    # Query de prueba
    test_query = "seismic design zone factor"
    
    # 1. Búsqueda con documentos existentes (modelo original)
    print("1. Búsqueda con embeddings existentes en la BD:")
    vector_store_existing = MongoVectorStoreRoBERTa(
        mongodb_uri=settings.mongodb_uri,
        database_name=settings.mongodb_database,
        collection_name=settings.mongodb_collection,
        use_roberta=True,  # Usamos RoBERTa para generar el query embedding
        embedding_model="roberta-base"
    )
    
    # Filtrar solo documentos con el modelo existente
    results_existing = vector_store_existing.similarity_search(
        query=test_query,
        top_k=3,
        filter_criteria={"model": "roberta-base"},  # O el modelo que esté en tu BD
        return_fields=["chunk_text", "similarity", "model", "dims"]
    )
    
    print(f"Query: '{test_query}'")
    if results_existing:
        for i, result in enumerate(results_existing, 1):
            print(f"  {i}. [{result.get('similarity', 0):.3f}] Modelo: {result.get('model')} "
                  f"(dims: {result.get('dims')})")
            print(f"     {result.get('chunk_text', '')[:100]}...")
    else:
        print("  No se encontraron resultados")
    
    # 2. Información sobre los modelos en la BD
    print("\n2. Modelos de embeddings en la base de datos:")
    stats = vector_store_existing.get_collection_stats()
    for model in stats.get('models_used', []):
        count = vector_store_existing.collection.count_documents({"model": model})
        print(f"  • {model}: {count} documentos")


def verify_document_structure():
    """
    Verifica la estructura de un documento aleatorio en la base de datos.
    """
    print("\n" + "="*60)
    print("VERIFICACIÓN: Estructura de Documento")
    print("="*60 + "\n")
    
    # Conectar a MongoDB
    from pymongo import MongoClient
    client = MongoClient(settings.mongodb_uri)
    db = client[settings.mongodb_database]
    collection = db[settings.mongodb_collection]
    
    # Obtener un documento de ejemplo
    sample_doc = collection.find_one({}, {"embedding": 0})  # Excluir embedding para visualización
    
    if sample_doc:
        print("Estructura del documento en MongoDB:")
        print("-" * 40)
        for key, value in sample_doc.items():
            if isinstance(value, str) and len(str(value)) > 100:
                print(f"  {key}: {str(value)[:100]}...")
            elif isinstance(value, dict):
                if "$date" in value:
                    print(f"  {key}: {value['$date']}")
                elif "$oid" in value:
                    print(f"  {key}: ObjectId('{value['$oid']}')")
                else:
                    print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
    else:
        print("No se encontraron documentos en la colección")
    
    client.close()


def main():
    """
    Función principal que ejecuta todas las pruebas.
    """
    print("\n" + "="*70)
    print(" PRUEBA DE INTEGRACIÓN CON BASE DE DATOS EXISTENTE ")
    print("="*70)
    print("\nEste script prueba la integración de RoBERTa con tu estructura")
    print("de documentos existente en MongoDB.\n")
    
    try:
        # 1. Verificar estructura de documentos
        print("1️⃣ Verificando estructura de documentos...")
        verify_document_structure()
        input("\nPresiona Enter para continuar...")
        
        # 2. Probar búsqueda en base de datos existente
        print("\n2️⃣ Probando búsqueda en base de datos existente...")
        test_similarity_search_existing_db()
        input("\nPresiona Enter para continuar...")
        
        # 3. Añadir nuevo documento con estructura correcta
        print("\n3️⃣ Añadiendo nuevo documento...")
        add_new_document_with_structure()
        input("\nPresiona Enter para continuar...")
        
        # 4. Comparar modelos
        print("\n4️⃣ Comparando modelos...")
        compare_models_on_existing_data()
        
        print("\n" + "="*70)
        print(" PRUEBAS COMPLETADAS ")
        print("="*70)
        print("\n✅ Todas las pruebas se ejecutaron correctamente.")
        print("📊 El sistema está adaptado para tu estructura de documentos.\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Pruebas interrumpidas por el usuario.")
    except Exception as e:
        logger.error(f"Error en las pruebas: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
