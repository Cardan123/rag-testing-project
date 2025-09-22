from typing import List, Dict, Any, Optional, Union
import numpy as np
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure
from sentence_transformers import SentenceTransformer
from .roberta_embeddings import RoBERTaEmbeddings
import logging

logger = logging.getLogger(__name__)


class MongoVectorStoreRoBERTa:
    """
    Vector store mejorado que soporta tanto RoBERTa como SentenceTransformers.
    """
    
    def __init__(
        self, 
        mongodb_uri: str, 
        database_name: str, 
        collection_name: str, 
        embedding_model: Optional[str] = None,
        use_roberta: bool = True,
        device: Optional[str] = None,
        batch_size: int = 8
    ):
        """
        Inicializa el vector store con MongoDB.
        
        Args:
            mongodb_uri: URI de conexión a MongoDB
            database_name: Nombre de la base de datos
            collection_name: Nombre de la colección
            embedding_model: Modelo de embeddings a usar
            use_roberta: Si usar RoBERTa (True) o SentenceTransformers (False)
            device: Dispositivo para el modelo
            batch_size: Tamaño del batch para procesamiento
        """
        # Conexión a MongoDB
        self.client = MongoClient(mongodb_uri)
        self.database = self.client[database_name]
        self.collection: Collection = self.database[collection_name]
        
        # Configurar modelo de embeddings
        self.use_roberta = use_roberta
        
        if use_roberta:
            model_name = embedding_model or "roberta-base"
            logger.info(f"Inicializando RoBERTa con modelo: {model_name}")
            self.embedding_model = RoBERTaEmbeddings(
                model_name=model_name,
                device=device,
                batch_size=batch_size
            )
        else:
            model_name = embedding_model or "all-MiniLM-L6-v2"
            logger.info(f"Inicializando SentenceTransformers con modelo: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)

        self._create_indexes()
        logger.info(f"MongoVectorStore inicializado con colección: {collection_name}")

    def _create_indexes(self):
        """Crea índices necesarios en MongoDB adaptados a la estructura existente."""
        try:
            # Índices para búsqueda eficiente según la estructura existente
            self.collection.create_index([("doc_path", 1)])
            self.collection.create_index([("chunk_index", 1)])
            self.collection.create_index([("model", 1)])
            self.collection.create_index([("chunk_method", 1)])
            self.collection.create_index([("created_at", -1)])
            # Índice compuesto para búsquedas eficientes
            self.collection.create_index([("doc_path", 1), ("chunk_index", 1)])
            logger.info("Índices creados exitosamente")
        except Exception as e:
            logger.warning(f"Error creando índices: {e}")

    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Añade documentos a la colección.
        
        Args:
            documents: Lista de documentos con embeddings
            
        Returns:
            True si la inserción fue exitosa
        """
        try:
            # Validar que los documentos tengan embeddings
            for doc in documents:
                if "embedding" not in doc:
                    logger.warning(f"Documento {doc.get('chunk_id', 'unknown')} sin embedding")
            
            # Insertar documentos
            result = self.collection.insert_many(documents, ordered=False)
            logger.info(f"Insertados {len(result.inserted_ids)} documentos")
            return True
            
        except Exception as e:
            logger.error(f"Error insertando documentos: {e}")
            return False

    def similarity_search(
        self, 
        query: str, 
        top_k: int = 5, 
        similarity_threshold: float = 0.7,
        filter_criteria: Optional[Dict[str, Any]] = None,
        return_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda por similitud usando embeddings.
        Adaptado para la estructura existente de documentos.
        
        Args:
            query: Texto de búsqueda
            top_k: Número de resultados a devolver
            similarity_threshold: Umbral mínimo de similitud
            filter_criteria: Criterios adicionales de filtrado
            return_fields: Campos específicos a devolver
            
        Returns:
            Lista de documentos similares ordenados por relevancia
        """
        # Generar embedding para la query
        if self.use_roberta:
            query_embedding = self.embedding_model.encode_queries(
                [query],  # RoBERTa espera una lista
                convert_to_tensor=False,
                normalize_embeddings=True
            )[0]  # Tomar el primer (y único) embedding
        else:
            query_embedding = self.embedding_model.encode([query])[0]
        
        # Convertir a lista si es necesario
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()
        
        # Verificar dimensión del embedding
        if len(query_embedding) != 768 and len(query_embedding) != 384:  # Dimensiones comunes
            logger.warning(f"Dimensión del query embedding: {len(query_embedding)}")
        
        # Construir pipeline de agregación
        pipeline = self._build_similarity_pipeline(
            query_embedding, 
            top_k, 
            similarity_threshold,
            filter_criteria,
            return_fields
        )

        try:
            results = list(self.collection.aggregate(pipeline, allowDiskUse=True))
            logger.info(f"Encontrados {len(results)} resultados con similitud >= {similarity_threshold}")
            return results
        except Exception as e:
            logger.error(f"Error en búsqueda de similitud: {e}")
            return []

    def _build_similarity_pipeline(
        self, 
        query_embedding: List[float], 
        top_k: int, 
        similarity_threshold: float,
        filter_criteria: Optional[Dict[str, Any]] = None,
        return_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Construye el pipeline de agregación para búsqueda de similitud.
        """
        pipeline = []
        
        # Añadir filtros adicionales si existen
        if filter_criteria:
            pipeline.append({"$match": filter_criteria})
        
        # Calcular similitud del coseno
        pipeline.extend([
            {
                "$addFields": {
                    "similarity": {
                        "$let": {
                            "vars": {
                                # Producto punto
                                "dot_product": {
                                    "$reduce": {
                                        "input": {"$range": [0, {"$size": "$embedding"}]},
                                        "initialValue": 0,
                                        "in": {
                                            "$add": [
                                                "$$value",
                                                {
                                                    "$multiply": [
                                                        {"$arrayElemAt": ["$embedding", "$$this"]},
                                                        {"$arrayElemAt": [query_embedding, "$$this"]}
                                                    ]
                                                }
                                            ]
                                        }
                                    }
                                },
                                # Magnitud del vector query
                                "query_magnitude": {
                                    "$sqrt": {
                                        "$reduce": {
                                            "input": query_embedding,
                                            "initialValue": 0,
                                            "in": {"$add": ["$$value", {"$multiply": ["$$this", "$$this"]}]}
                                        }
                                    }
                                },
                                # Magnitud del vector documento
                                "doc_magnitude": {
                                    "$sqrt": {
                                        "$reduce": {
                                            "input": "$embedding",
                                            "initialValue": 0,
                                            "in": {"$add": ["$$value", {"$multiply": ["$$this", "$$this"]}]}
                                        }
                                    }
                                }
                            },
                            "in": {
                                "$cond": {
                                    "if": {
                                        "$and": [
                                            {"$gt": ["$$query_magnitude", 0]},
                                            {"$gt": ["$$doc_magnitude", 0]}
                                        ]
                                    },
                                    "then": {
                                        "$divide": [
                                            "$$dot_product",
                                            {"$multiply": ["$$query_magnitude", "$$doc_magnitude"]}
                                        ]
                                    },
                                    "else": 0
                                }
                            }
                        }
                    }
                }
            },
            # Filtrar por umbral de similitud
            {"$match": {"similarity": {"$gte": similarity_threshold}}},
            # Ordenar por similitud
            {"$sort": {"similarity": -1}},
            # Limitar resultados
            {"$limit": top_k},
            # Proyectar campos según la estructura existente
            {
                "$project": self._get_projection_fields(return_fields)
            }
        ])
        
        return pipeline
    
    def _get_projection_fields(self, return_fields: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Construye el objeto de proyección para MongoDB.
        
        Args:
            return_fields: Lista de campos a incluir en el resultado
            
        Returns:
            Diccionario de proyección para MongoDB
        """
        if return_fields:
            projection = {field: 1 for field in return_fields}
            # Incluir _id si no está especificado
            if "_id" not in projection:
                projection["_id"] = 1
            # NO incluir embedding explícitamente (MongoDB no permite mezclar inclusión con exclusión)
            # El embedding simplemente no se incluirá si no está en return_fields
        else:
            # Proyección por defecto - excluir solo el embedding
            projection = {
                "embedding": 0
            }
        return projection

    def hybrid_search(
        self,
        query: str,
        text_search_field: str = "content",
        top_k: int = 5,
        similarity_weight: float = 0.7,
        text_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda híbrida que combina similitud de embeddings con búsqueda de texto.
        
        Args:
            query: Texto de búsqueda
            text_search_field: Campo para búsqueda de texto
            top_k: Número de resultados
            similarity_weight: Peso para similitud de embeddings
            text_weight: Peso para búsqueda de texto
            
        Returns:
            Lista de resultados combinados
        """
        # Búsqueda por similitud
        similarity_results = self.similarity_search(query, top_k=top_k*2)
        
        # Búsqueda de texto
        text_results = list(self.collection.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}, "embedding": 0}
        ).sort([("score", {"$meta": "textScore"})]).limit(top_k*2))
        
        # Combinar y ponderar resultados
        combined_results = {}
        
        # Procesar resultados de similitud
        for result in similarity_results:
            chunk_id = result["chunk_id"]
            combined_results[chunk_id] = {
                **result,
                "final_score": result["similarity"] * similarity_weight
            }
        
        # Procesar resultados de texto
        max_text_score = max([r.get("score", 0) for r in text_results], default=1)
        for result in text_results:
            chunk_id = result["chunk_id"]
            normalized_score = result.get("score", 0) / max_text_score if max_text_score > 0 else 0
            
            if chunk_id in combined_results:
                combined_results[chunk_id]["final_score"] += normalized_score * text_weight
            else:
                result["final_score"] = normalized_score * text_weight
                combined_results[chunk_id] = result
        
        # Ordenar por puntuación final y devolver top_k
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x["final_score"],
            reverse=True
        )[:top_k]
        
        return sorted_results

    def get_document_by_path(self, doc_path: str, chunk_index: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Obtiene un documento por su path y opcionalmente por chunk_index."""
        query = {"doc_path": doc_path}
        if chunk_index is not None:
            query["chunk_index"] = chunk_index
        return self.collection.find_one(query, {"embedding": 0})
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene un documento por su ID (compatibilidad hacia atrás)."""
        # Intenta buscar por doc_id si existe, o por _id
        return self.collection.find_one(
            {"$or": [{"doc_id": doc_id}, {"_id": doc_id}]}, 
            {"embedding": 0}
        )

    def get_documents_by_ids(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """Obtiene múltiples documentos por sus IDs."""
        return list(self.collection.find(
            {"doc_id": {"$in": doc_ids}},
            {"embedding": 0}
        ))

    def delete_documents_by_path(self, doc_paths: List[str]) -> int:
        """Elimina documentos por sus paths."""
        result = self.collection.delete_many({"doc_path": {"$in": doc_paths}})
        logger.info(f"Eliminados {result.deleted_count} documentos")
        return result.deleted_count
    
    def delete_documents(self, doc_ids: List[str]) -> int:
        """Elimina documentos por sus IDs (compatibilidad hacia atrás)."""
        result = self.collection.delete_many(
            {"$or": [
                {"doc_id": {"$in": doc_ids}},
                {"_id": {"$in": doc_ids}}
            ]}
        )
        logger.info(f"Eliminados {result.deleted_count} documentos")
        return result.deleted_count

    def update_document_embedding(self, doc_path: str, chunk_index: int, new_embedding: List[float]) -> bool:
        """Actualiza el embedding de un documento específico."""
        try:
            from datetime import datetime
            result = self.collection.update_one(
                {"doc_path": doc_path, "chunk_index": chunk_index},
                {"$set": {
                    "embedding": new_embedding,
                    "dims": len(new_embedding),
                    "model": self.embedding_model_name if hasattr(self, 'embedding_model_name') else "updated",
                    "updated_at": datetime.utcnow()
                }}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error actualizando embedding: {e}")
            return False

    def clear_collection(self):
        """Limpia toda la colección."""
        result = self.collection.delete_many({})
        logger.info(f"Colección limpiada. {result.deleted_count} documentos eliminados")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la colección adaptadas a la estructura existente."""
        stats = {
            "total_documents": self.collection.count_documents({}),
            "unique_doc_paths": len(self.collection.distinct("doc_path")),
            "models_used": self.collection.distinct("model"),
            "chunk_methods": self.collection.distinct("chunk_method"),
            "embedding_dimensions": self.collection.distinct("dims"),
            "average_chunk_size": 0
        }
        
        # Calcular tamaño promedio de chunks
        pipeline = [
            {"$group": {
                "_id": None,
                "avg_size": {"$avg": "$metadata.chunk_size"}
            }}
        ]
        
        result = list(self.collection.aggregate(pipeline))
        if result:
            stats["average_chunk_size"] = result[0].get("avg_size", 0)
        
        return stats

    def health_check(self) -> bool:
        """Verifica la salud de la conexión a MongoDB."""
        try:
            self.client.admin.command('ping')
            return True
        except ConnectionFailure:
            logger.error("Fallo en health check de MongoDB")
            return False
    
    def create_text_index(self, fields: List[str]):
        """
        Crea un índice de texto para búsqueda híbrida.
        
        Args:
            fields: Lista de campos para indexar
        """
        try:
            index_spec = [(field, "text") for field in fields]
            self.collection.create_index(index_spec)
            logger.info(f"Índice de texto creado para campos: {fields}")
        except Exception as e:
            logger.error(f"Error creando índice de texto: {e}")
