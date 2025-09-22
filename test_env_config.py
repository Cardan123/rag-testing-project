"""
Script para verificar la lectura del archivo .env
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Cargar .env manualmente
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path, verbose=True, override=True)

print("=" * 60)
print("VERIFICACIÓN DE VARIABLES DE ENTORNO")
print("=" * 60)

# Variables esperadas en .env
env_vars = [
    "GEMINI_API_KEY",
    "MONGODB_URI", 
    "MONGODB_DATABASE",
    "MONGODB_COLLECTION",
    "USE_ROBERTA",
    "EMBEDDING_MODEL",
    "SENTENCE_TRANSFORMER_MODEL",
    "GEMINI_MODEL",
    "ROBERTA_DEVICE",
    "ROBERTA_BATCH_SIZE",
    "ROBERTA_MAX_LENGTH",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K_RETRIEVAL"
]

print("\n1. Variables de entorno cargadas directamente con os.environ:")
print("-" * 40)
for var in env_vars:
    value = os.environ.get(var)
    if value:
        # Ocultar API keys
        if "API_KEY" in var:
            display_value = value[:10] + "..." if len(value) > 10 else value
        else:
            display_value = value
        print(f"✓ {var}: {display_value}")
    else:
        print(f"✗ {var}: No definida")

print("\n2. Verificando con pydantic Settings:")
print("-" * 40)

try:
    from src.config import settings
    
    print("\nValores en settings:")
    # Verificar valores clave
    attrs = [
        ('gemini_api_key', 'API Key'),
        ('mongodb_uri', 'MongoDB URI'),
        ('mongodb_database', 'MongoDB Database'),
        ('mongodb_collection', 'MongoDB Collection'),
        ('use_roberta', 'Use RoBERTa'),
        ('embedding_model', 'Embedding Model'),
        ('sentence_transformer_model', 'Sentence Transformer'),
        ('roberta_device', 'RoBERTa Device'),
        ('roberta_batch_size', 'Batch Size'),
        ('chunk_size', 'Chunk Size'),
        ('chunk_overlap', 'Chunk Overlap'),
        ('top_k_retrieval', 'Top K'),
    ]
    
    for attr, name in attrs:
        if hasattr(settings, attr):
            value = getattr(settings, attr)
            if 'api_key' in attr.lower() and value:
                display_value = str(value)[:10] + "..." if len(str(value)) > 10 else str(value)
            else:
                display_value = value
            print(f"  • {name}: {display_value}")
        else:
            print(f"  • {name}: Atributo no encontrado")
            
except Exception as e:
    print(f"Error al importar settings: {e}")

print("\n3. Verificando archivo .env:")
print("-" * 40)

if env_path.exists():
    print(f"✓ Archivo .env encontrado en: {env_path.absolute()}")
    with open(env_path, 'r') as f:
        lines = f.readlines()
    print(f"  Contiene {len(lines)} líneas")
    # Contar variables definidas
    var_count = sum(1 for line in lines if '=' in line and not line.strip().startswith('#'))
    print(f"  Variables definidas: {var_count}")
else:
    print(f"✗ Archivo .env NO encontrado en: {env_path.absolute()}")
    print("\n  Creando archivo .env desde .env.example...")
    
    example_path = Path('.') / '.env.example'
    if example_path.exists():
        import shutil
        shutil.copy(example_path, env_path)
        print("  ✓ Archivo .env creado desde .env.example")
        print("  ⚠️ Recuerda actualizar las API keys y configuraciones")
    else:
        print("  ✗ No se encontró .env.example")

print("\n4. Diagnóstico de problemas comunes:")
print("-" * 40)

# Verificar si está usando el archivo correcto
import sys
print(f"• Python Path: {sys.executable}")
print(f"• Working Directory: {Path.cwd()}")

# Verificar si load_dotenv funciona
from dotenv import dotenv_values
config = dotenv_values(".env")
print(f"• Variables cargadas con dotenv_values: {len(config)}")

if len(config) == 0:
    print("\n⚠️ PROBLEMA DETECTADO: No se están cargando variables del .env")
    print("  Posibles soluciones:")
    print("  1. Verifica que el archivo .env existe y no está vacío")
    print("  2. Asegúrate de que el archivo .env tiene el formato correcto (VAR=valor)")
    print("  3. Verifica que no haya espacios antes del nombre de la variable")
else:
    print(f"\n✓ dotenv está funcionando correctamente")
    
print("\n" + "=" * 60)
