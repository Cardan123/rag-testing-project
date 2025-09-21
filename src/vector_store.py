from typing import List, Dict, Any, Optional
import numpy as np
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class MongoVectorStore:
    def __init__(self, mongodb_uri: str, database_name: str, collection_name: str, embedding_model: str):
        self.client = MongoClient(mongodb_uri)
        self.database = self.client[database_name]
        self.collection: Collection = self.database[collection_name]
        self.embedding_model = SentenceTransformer(embedding_model)

        self._create_indexes()

    def _create_indexes(self):
        try:
            self.collection.create_index([("doc_id", 1)])
            self.collection.create_index([("chunk_id", 1)], unique=True)
            logger.info("Indexes created successfully")
        except Exception as e:
            logger.warning(f"Index creation failed: {e}")

    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        try:
            result = self.collection.insert_many(documents, ordered=False)
            logger.info(f"Inserted {len(result.inserted_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            return False

    def similarity_search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode([query])[0].tolist()

        pipeline = [
            {
                "$addFields": {
                    "similarity": {
                        "$let": {
                            "vars": {
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
                                "query_magnitude": {
                                    "$sqrt": {
                                        "$reduce": {
                                            "input": query_embedding,
                                            "initialValue": 0,
                                            "in": {"$add": ["$$value", {"$multiply": ["$$this", "$$this"]}]}
                                        }
                                    }
                                },
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
                                "$divide": [
                                    "$$dot_product",
                                    {"$multiply": ["$$query_magnitude", "$$doc_magnitude"]}
                                ]
                            }
                        }
                    }
                }
            },
            {"$match": {"similarity": {"$gte": similarity_threshold}}},
            {"$sort": {"similarity": -1}},
            {"$limit": top_k},
            {"$project": {"embedding": 0}}
        ]

        try:
            results = list(self.collection.aggregate(pipeline))
            return results
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self.collection.find_one({"doc_id": doc_id}, {"embedding": 0})

    def delete_documents(self, doc_ids: List[str]) -> int:
        result = self.collection.delete_many({"doc_id": {"$in": doc_ids}})
        return result.deleted_count

    def clear_collection(self):
        self.collection.delete_many({})

    def get_collection_stats(self) -> Dict[str, Any]:
        return {
            "total_documents": self.collection.count_documents({}),
            "unique_doc_ids": len(self.collection.distinct("doc_id"))
        }

    def health_check(self) -> bool:
        try:
            self.client.admin.command('ping')
            return True
        except ConnectionFailure:
            return False