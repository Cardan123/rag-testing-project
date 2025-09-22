"""
Script para verificar la conexión a MongoDB Atlas
"""

from pymongo import MongoClient
from src.config import settings
import sys

print("=" * 60)
print("PRUEBA DE CONEXIÓN A MONGODB ATLAS")
print("=" * 60)

print(f"\n📌 Configuración:")
print(f"  • URI: {settings.mongodb_uri[:40]}...")
print(f"  • Database: {settings.mongodb_database}")
print(f"  • Collection: {settings.mongodb_collection}")

print("\n🔌 Intentando conectar a MongoDB Atlas...")

try:
    # Conectar a MongoDB
    client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)
    
    # Forzar una conexión para verificar
    client.admin.command('ping')
    print("✅ Conexión exitosa a MongoDB Atlas!")
    
    # Obtener información de la base de datos
    db = client[settings.mongodb_database]
    collection = db[settings.mongodb_collection]
    
    # Contar documentos
    doc_count = collection.count_documents({})
    print(f"\n📊 Estadísticas de la colección '{settings.mongodb_collection}':")
    print(f"  • Total de documentos: {doc_count}")
    
    # Obtener información sobre los modelos usados
    models = collection.distinct("model")
    if models:
        print(f"  • Modelos de embeddings encontrados: {models}")
    
    # Obtener un documento de muestra (sin embedding)
    sample = collection.find_one({}, {"embedding": 0})
    if sample:
        print("\n📄 Estructura de documento encontrada:")
        for key in sample.keys():
            if key != "_id":
                print(f"    - {key}")
    
    # Verificar dimensiones de embeddings
    dims = collection.distinct("dims")
    if dims:
        print(f"\n📏 Dimensiones de embeddings: {dims}")
    
    print("\n✅ Tu configuración está correcta y la base de datos es accesible!")
    
except Exception as e:
    print(f"\n❌ Error al conectar: {e}")
    print("\n💡 Posibles soluciones:")
    print("  1. Verifica que tu IP esté en la whitelist de MongoDB Atlas")
    print("  2. Verifica las credenciales en el URI")
    print("  3. Asegúrate de que el cluster esté activo")
    sys.exit(1)
finally:
    if 'client' in locals():
        client.close()

print("\n" + "=" * 60)
