"""
Script para buscar en tu base de datos MongoDB Atlas con RoBERTa
"""

from src.vector_store_roberta import MongoVectorStoreRoBERTa
from src.config import settings
import logging

logging.basicConfig(level=logging.INFO)

print("\n" + "="*60)
print("BÚSQUEDA EN TU BASE DE DATOS MONGODB ATLAS")
print("="*60 + "\n")

# Inicializar vector store con tu configuración
print("🔌 Conectando a MongoDB Atlas...")
vector_store = MongoVectorStoreRoBERTa(
    mongodb_uri=settings.mongodb_uri,
    database_name=settings.mongodb_database,
    collection_name=settings.mongodb_collection,
    use_roberta=settings.use_roberta,
    embedding_model=settings.embedding_model,
    device=settings.roberta_device,
    batch_size=settings.roberta_batch_size
)

# Verificar estadísticas
stats = vector_store.get_collection_stats()
print(f"\n📊 Estadísticas de tu colección:")
print(f"  • Total documentos: {stats['total_documents']}")
print(f"  • Documentos únicos: {stats['unique_doc_paths']}")
print(f"  • Modelos usados: {stats['models_used']}")

# Realizar búsquedas de ejemplo
queries = [
    "concrete cover requirements",
    "seismic design parameters zone factor",
    "reduction coefficient for structural design",
    "live loads for basements and floors"
]

print("\n🔍 Realizando búsquedas de ejemplo:")
print("-" * 40)

for query in queries:
    print(f"\nQuery: '{query}'")
    
    # Búsqueda con tu configuración
    results = vector_store.similarity_search(
        query=query,
        top_k=3,
        similarity_threshold=0.3,
        return_fields=["doc_path", "chunk_text", "chunk_index", "similarity"]
    )
    
    if results:
        print(f"  Encontrados {len(results)} resultados:")
        for i, result in enumerate(results, 1):
            # Extraer solo el nombre del archivo
            doc_name = result.get("doc_path", "").split("/")[-1] if "/" in result.get("doc_path", "") else result.get("doc_path", "Unknown")
            chunk_idx = result.get("chunk_index", "?")
            similarity = result.get("similarity", 0)
            chunk_preview = result.get("chunk_text", "")[:100].replace("\n", " ")
            
            print(f"    {i}. [{similarity:.3f}] {doc_name} (chunk {chunk_idx})")
            print(f"       {chunk_preview}...")
    else:
        print("    No se encontraron resultados")

print("\n✅ Sistema funcionando correctamente con tu base de datos!")
print("=" * 60)
