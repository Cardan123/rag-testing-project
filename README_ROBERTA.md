# 🤖 Integración de RoBERTa en el Sistema RAG

Este documento explica cómo usar **RoBERTa-base** como modelo de embeddings en el sistema RAG.

## 📋 Tabla de Contenidos
- [Características](#características)
- [Instalación](#instalación)
- [Configuración](#configuración)
- [Uso](#uso)
- [Comparación con SentenceTransformers](#comparación-con-sentencetransformers)
- [Ejemplos de Código](#ejemplos-de-código)
- [Consideraciones de Rendimiento](#consideraciones-de-rendimiento)

## ✨ Características

### RoBERTa vs BERT
RoBERTa (Robustly Optimized BERT Pretraining Approach) es una versión mejorada de BERT que:
- **Entrena más tiempo** con batches más grandes
- **Elimina NSP** (Next Sentence Prediction)
- **Usa máscaras dinámicas** que cambian en cada época
- **Entrena con más datos** (160GB vs 16GB de BERT)

### Ventajas en RAG
- **Mejor comprensión contextual**: Captura relaciones semánticas más profundas
- **Embeddings de alta calidad**: 768 dimensiones (base) o 1024 (large)
- **Multilingüe**: Soporta múltiples idiomas
- **Transfer learning**: Se adapta bien a dominios específicos

## 🚀 Instalación

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

Las nuevas dependencias incluyen:
- `transformers>=4.30.0` - Biblioteca de Hugging Face
- `torch>=2.0.0` - PyTorch para el modelo

### 2. Configurar variables de entorno
Copia el archivo de ejemplo y edítalo:
```bash
cp .env.example .env
```

Configura las variables para RoBERTa:
```bash
# Activar RoBERTa
USE_ROBERTA=true
EMBEDDING_MODEL=roberta-base

# Configuración opcional
ROBERTA_DEVICE=cuda  # Si tienes GPU
ROBERTA_BATCH_SIZE=16  # Aumentar si tienes más memoria
ROBERTA_MAX_LENGTH=512  # Longitud máxima de tokens
```

## ⚙️ Configuración

### Modelos Disponibles
| Modelo | Parámetros | Dimensión | Uso Recomendado |
|--------|------------|-----------|-----------------|
| `roberta-base` | 125M | 768 | Balance velocidad/calidad |
| `roberta-large` | 355M | 1024 | Máxima calidad |
| `distilroberta-base` | 82M | 768 | Mayor velocidad |
| `xlm-roberta-base` | 270M | 768 | Multilingüe |

### Configuración en Python
```python
from src.config import settings

# Las configuraciones se cargan automáticamente desde .env
print(f"Usando RoBERTa: {settings.use_roberta}")
print(f"Modelo: {settings.embedding_model}")
print(f"Device: {settings.roberta_device or 'auto'}")
```

## 📖 Uso

### 1. Procesamiento de Documentos con RoBERTa

```python
from src.document_processor_roberta import DocumentProcessorRoBERTa

# Inicializar procesador con RoBERTa
processor = DocumentProcessorRoBERTa(
    use_roberta=True,
    embedding_model="roberta-base",
    chunk_size=500,
    chunk_overlap=100,
    device="cuda"  # o "cpu"
)

# Procesar un texto
text = "Tu texto aquí..."
chunks = processor.process_single_text(text)

# Procesar documentos Markdown
all_chunks = processor.process_documents(
    "ruta/a/documentos",
    file_pattern="*.md",
    show_progress=True
)
```

### 2. Vector Store con RoBERTa

```python
from src.vector_store_roberta import MongoVectorStoreRoBERTa

# Inicializar vector store
vector_store = MongoVectorStoreRoBERTa(
    mongodb_uri="mongodb://localhost:27017",
    database_name="rag_testing",
    collection_name="roberta_embeddings",
    use_roberta=True,
    embedding_model="roberta-base"
)

# Almacenar documentos
vector_store.add_documents(chunks)

# Búsqueda por similitud
results = vector_store.similarity_search(
    query="¿Qué es machine learning?",
    top_k=5,
    similarity_threshold=0.7
)
```

### 3. Uso Directo de RoBERTa Embeddings

```python
from src.roberta_embeddings import RoBERTaEmbeddings

# Inicializar modelo
embeddings_model = RoBERTaEmbeddings(
    model_name="roberta-base",
    device="cuda",
    batch_size=8
)

# Generar embeddings para documentos
docs = ["Documento 1", "Documento 2", "Documento 3"]
doc_embeddings = embeddings_model.encode_documents(
    docs,
    normalize_embeddings=True,
    show_progress_bar=True
)

# Generar embeddings para queries
query = "Mi pregunta de búsqueda"
query_embedding = embeddings_model.encode_queries(query)

# Calcular similitud
similarities = embeddings_model.similarity(query_embedding, doc_embeddings)
```

## 📊 Comparación con SentenceTransformers

| Aspecto | RoBERTa-base | SentenceTransformers |
|---------|--------------|----------------------|
| **Dimensión de embeddings** | 768 | 384-768 |
| **Velocidad** | Medio | Rápido |
| **Calidad semántica** | Excelente | Muy buena |
| **Uso de memoria** | ~500MB | ~100-400MB |
| **Comprensión contextual** | Superior | Buena |
| **Optimizado para similitud** | No | Sí |

### ¿Cuándo usar cada uno?

**Usa RoBERTa cuando:**
- Necesitas máxima calidad en comprensión semántica
- Trabajas con textos complejos o técnicos
- Tienes recursos computacionales disponibles (GPU)
- La precisión es más importante que la velocidad

**Usa SentenceTransformers cuando:**
- Necesitas procesamiento rápido
- Trabajas con grandes volúmenes de datos
- Los recursos son limitados
- La tarea principal es búsqueda por similitud

## 💻 Ejemplos de Código

### Ejemplo Completo: Sistema RAG con RoBERTa

```python
import os
from dotenv import load_dotenv
from src.document_processor_roberta import DocumentProcessorRoBERTa
from src.vector_store_roberta import MongoVectorStoreRoBERTa

# Cargar configuración
load_dotenv()

def setup_rag_system():
    """Configura el sistema RAG completo con RoBERTa."""
    
    # 1. Procesar documentos
    processor = DocumentProcessorRoBERTa(
        use_roberta=True,
        embedding_model="roberta-base",
        chunk_size=500
    )
    
    chunks = processor.process_documents("./documents")
    
    # 2. Almacenar en MongoDB
    vector_store = MongoVectorStoreRoBERTa(
        mongodb_uri=os.getenv("MONGODB_URI"),
        database_name="rag_roberta",
        collection_name="documents",
        use_roberta=True
    )
    
    vector_store.add_documents(chunks)
    
    # 3. Función de búsqueda
    def search(query: str, top_k: int = 5):
        results = vector_store.similarity_search(
            query=query,
            top_k=top_k,
            similarity_threshold=0.6
        )
        return results
    
    return search

# Usar el sistema
search_fn = setup_rag_system()
results = search_fn("¿Cómo funciona RoBERTa?")
for result in results:
    print(f"[{result['similarity']:.3f}] {result['content'][:100]}...")
```

### Búsqueda Híbrida

```python
# Crear índice de texto para búsqueda híbrida
vector_store.create_text_index(["content", "metadata.title"])

# Realizar búsqueda híbrida
results = vector_store.hybrid_search(
    query="machine learning con transformers",
    text_search_field="content",
    top_k=10,
    similarity_weight=0.7,  # 70% peso para embeddings
    text_weight=0.3         # 30% peso para búsqueda de texto
)
```

## ⚡ Consideraciones de Rendimiento

### Optimización de Memoria
```python
# Para GPUs con memoria limitada
processor = DocumentProcessorRoBERTa(
    use_roberta=True,
    batch_size=4,  # Reducir batch size
    device="cuda"
)
```

### Procesamiento en Lotes
```python
# Procesar documentos grandes en lotes
import numpy as np

docs = ["doc1", "doc2", ..., "doc1000"]
batch_size = 100

all_embeddings = []
for i in range(0, len(docs), batch_size):
    batch = docs[i:i+batch_size]
    embeddings = embeddings_model.encode_documents(batch)
    all_embeddings.append(embeddings)

final_embeddings = np.vstack(all_embeddings)
```

### Cache de Embeddings
```python
import pickle

# Guardar embeddings
with open('embeddings_cache.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

# Cargar embeddings
with open('embeddings_cache.pkl', 'rb') as f:
    cached_embeddings = pickle.load(f)
```

## 🧪 Testing

Ejecuta el script de ejemplo para probar todas las funcionalidades:

```bash
python example_roberta_usage.py
```

Este script incluye:
- Demo de procesamiento con RoBERTa
- Almacenamiento en MongoDB
- Búsquedas por similitud
- Comparación con SentenceTransformers
- Procesamiento de documentos Markdown

## 📈 Métricas y Evaluación

Para evaluar la calidad de los embeddings:

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Evaluar coherencia semántica
test_pairs = [
    ("machine learning", "aprendizaje automático"),
    ("deep learning", "redes neuronales profundas"),
    ("gato", "perro"),  # Relacionados pero diferentes
    ("computadora", "pizza")  # No relacionados
]

for text1, text2 in test_pairs:
    emb1 = embeddings_model.encode([text1])
    emb2 = embeddings_model.encode([text2])
    sim = cosine_similarity(emb1, emb2)[0][0]
    print(f"{text1} <-> {text2}: {sim:.3f}")
```

## 🔧 Troubleshooting

### Error: CUDA out of memory
```python
# Solución 1: Reducir batch size
processor = DocumentProcessorRoBERTa(batch_size=2)

# Solución 2: Usar CPU
processor = DocumentProcessorRoBERTa(device="cpu")

# Solución 3: Usar modelo más pequeño
processor = DocumentProcessorRoBERTa(
    embedding_model="distilroberta-base"
)
```

### Error: Modelo descarga lenta
```python
# Usar mirror de Hugging Face
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

### Embeddings no normalizados
```python
# Siempre normalizar para cosine similarity
embeddings = embeddings_model.encode(
    texts,
    normalize_embeddings=True  # Importante!
)
```

## 📚 Referencias

- [RoBERTa Paper](https://arxiv.org/abs/1907.11692)
- [Hugging Face RoBERTa](https://huggingface.co/roberta-base)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Transformers Library](https://huggingface.co/docs/transformers)

## 🤝 Soporte

Si tienes problemas o preguntas:
1. Revisa este README
2. Ejecuta el script de ejemplo: `python example_roberta_usage.py`
3. Verifica los logs en la consola
4. Asegúrate de que MongoDB esté ejecutándose

---

**Nota:** RoBERTa requiere más recursos computacionales que SentenceTransformers. Para producción, considera usar GPU o un servicio de inferencia como Hugging Face Inference API.
