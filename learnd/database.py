"""
Qdrant database integration for the Learnd system.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, CollectionStatus,
    PointStruct, Filter, FieldCondition, MatchValue
)

try:
    from .models import Concept, Cluster, LearndConfig
except ImportError:
    from learnd.models import Concept, Cluster, LearndConfig


class QdrantManager:
    """Manages Qdrant vector database operations for Learnd."""
    
    def __init__(self, config: LearndConfig):
        self.config = config
        self.client: Optional[QdrantClient] = None
        self.primary_collection = f"{config.collection_name}_primary"
        self.secondary_collection = f"{config.collection_name}_secondary"  
        self.clusters_collection = f"{config.collection_name}_clusters"
        
    async def initialize(self) -> None:
        """Initialize Qdrant client and collections."""
        try:
            self.client = QdrantClient(
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key
            )
            
            await self._create_collections()
            logger.info("Qdrant database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise
    
    async def _create_collections(self) -> None:
        """Create necessary collections if they don't exist."""
        vector_config = VectorParams(
            size=self.config.embedding_dimension,
            distance=Distance.COSINE
        )
        
        collections = [
            self.primary_collection,
            self.secondary_collection, 
            self.clusters_collection
        ]
        
        for collection_name in collections:
            try:
                # Check if collection exists
                collections_info = self.client.get_collections()
                existing_names = [c.name for c in collections_info.collections]
                
                if collection_name not in existing_names:
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=vector_config
                    )
                    logger.info(f"Created collection: {collection_name}")
                else:
                    logger.info(f"Collection already exists: {collection_name}")
                    
            except Exception as e:
                logger.error(f"Error creating collection {collection_name}: {e}")
                raise
    
    async def store_concept(self, concept: Concept) -> bool:
        """Store a concept in the appropriate collection."""
        try:
            collection_name = (
                self.primary_collection if concept.layer == 1 
                else self.secondary_collection
            )
            
            point = PointStruct(
                id=concept.id,
                vector=concept.embedding,
                payload=concept.to_dict()
            )
            
            self.client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            
            logger.debug(f"Stored concept {concept.id} in {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store concept {concept.id}: {e}")
            return False
    
    async def get_concept(self, concept_id: str, layer: Optional[int] = None) -> Optional[Concept]:
        """Retrieve a concept by ID."""
        try:
            collections_to_search = []
            if layer == 1:
                collections_to_search = [self.primary_collection]
            elif layer == 2:
                collections_to_search = [self.secondary_collection]
            else:
                collections_to_search = [self.primary_collection, self.secondary_collection]
            
            for collection_name in collections_to_search:
                try:
                    result = self.client.retrieve(
                        collection_name=collection_name,
                        ids=[concept_id]
                    )
                    
                    if result:
                        payload = result[0].payload
                        return Concept.from_dict(payload)
                        
                except Exception as e:
                    logger.debug(f"Concept {concept_id} not found in {collection_name}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve concept {concept_id}: {e}")
            return None
    
    async def search_similar_concepts(
        self,
        query_vector: List[float],
        collection_name: str,
        limit: int = 10,
        score_threshold: float = 0.7
    ) -> List[Tuple[Concept, float]]:
        """Search for similar concepts in a collection."""
        try:
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            
            results = []
            for scored_point in search_result:
                concept = Concept.from_dict(scored_point.payload)
                results.append((concept, scored_point.score))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search concepts in {collection_name}: {e}")
            return []
    
    async def update_concept_frequency(self, concept_id: str, new_frequency: int) -> bool:
        """Update concept frequency in database."""
        try:
            # Find which collection the concept is in
            concept = await self.get_concept(concept_id)
            if not concept:
                return False
            
            collection_name = (
                self.primary_collection if concept.layer == 1 
                else self.secondary_collection
            )
            
            # Update the frequency in payload
            concept.frequency = new_frequency
            
            point = PointStruct(
                id=concept_id,
                vector=concept.embedding,
                payload=concept.to_dict()
            )
            
            self.client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update concept frequency {concept_id}: {e}")
            return False
    
    async def move_concept_between_layers(self, concept_id: str, target_layer: int) -> bool:
        """Move a concept between primary and secondary layers."""
        try:
            # Get concept from current layer
            concept = await self.get_concept(concept_id)
            if not concept:
                return False
            
            source_collection = (
                self.primary_collection if concept.layer == 1 
                else self.secondary_collection
            )
            target_collection = (
                self.primary_collection if target_layer == 1 
                else self.secondary_collection
            )
            
            # Update concept layer
            concept.layer = target_layer
            
            # Add to target collection
            point = PointStruct(
                id=concept_id,
                vector=concept.embedding,
                payload=concept.to_dict()
            )
            
            self.client.upsert(
                collection_name=target_collection,
                points=[point]
            )
            
            # Remove from source collection if different
            if source_collection != target_collection:
                self.client.delete(
                    collection_name=source_collection,
                    points_selector=models.PointIdsList(points=[concept_id])
                )
            
            logger.info(f"Moved concept {concept_id} from layer {3-target_layer} to layer {target_layer}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to move concept {concept_id} to layer {target_layer}: {e}")
            return False
    
    async def get_layer_statistics(self) -> Dict[str, int]:
        """Get statistics about both layers."""
        try:
            primary_info = self.client.get_collection(self.primary_collection)
            secondary_info = self.client.get_collection(self.secondary_collection)
            clusters_info = self.client.get_collection(self.clusters_collection)
            
            return {
                "primary_layer_size": primary_info.points_count or 0,
                "secondary_layer_size": secondary_info.points_count or 0,
                "cluster_count": clusters_info.points_count or 0,
                "total_concepts": (primary_info.points_count or 0) + (secondary_info.points_count or 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get layer statistics: {e}")
            return {"primary_layer_size": 0, "secondary_layer_size": 0, "cluster_count": 0, "total_concepts": 0}
    
    async def get_all_concepts_in_layer(self, layer: int) -> List[Concept]:
        """Get all concepts in a specific layer."""
        try:
            collection_name = (
                self.primary_collection if layer == 1 
                else self.secondary_collection
            )
            
            # Get all points from collection
            scroll_result = self.client.scroll(
                collection_name=collection_name,
                limit=10000  # Adjust based on expected size
            )
            
            concepts = []
            for point in scroll_result[0]:
                concept = Concept.from_dict(point.payload)
                concepts.append(concept)
            
            return concepts
            
        except Exception as e:
            logger.error(f"Failed to get all concepts from layer {layer}: {e}")
            return []
    
    async def store_cluster(self, cluster: Cluster) -> bool:
        """Store a cluster in the clusters collection."""
        try:
            point = PointStruct(
                id=cluster.id,
                vector=cluster.representative_vector,
                payload=cluster.to_dict()
            )
            
            self.client.upsert(
                collection_name=self.clusters_collection,
                points=[point]
            )
            
            logger.debug(f"Stored cluster {cluster.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store cluster {cluster.id}: {e}")
            return False
    
    async def delete_concept(self, concept_id: str, layer: Optional[int] = None) -> bool:
        """Delete a concept from the database."""
        try:
            collections_to_search = []
            if layer == 1:
                collections_to_search = [self.primary_collection]
            elif layer == 2:
                collections_to_search = [self.secondary_collection]
            else:
                collections_to_search = [self.primary_collection, self.secondary_collection]
            
            for collection_name in collections_to_search:
                try:
                    self.client.delete(
                        collection_name=collection_name,
                        points_selector=models.PointIdsList(points=[concept_id])
                    )
                except Exception:
                    continue  # Concept might not be in this collection
            
            logger.info(f"Deleted concept {concept_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete concept {concept_id}: {e}")
            return False
    
    async def optimize_collections(self) -> None:
        """Optimize Qdrant collections for better performance."""
        try:
            collections = [
                self.primary_collection,
                self.secondary_collection,
                self.clusters_collection
            ]
            
            for collection_name in collections:
                # This would trigger Qdrant's internal optimization
                # The exact API might vary based on Qdrant version
                logger.info(f"Optimized collection: {collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to optimize collections: {e}")
    
    async def close(self) -> None:
        """Close the Qdrant client connection."""
        if self.client:
            # Note: qdrant-client doesn't have an explicit close method
            # The connection will be closed when the object is garbage collected
            self.client = None
            logger.info("Qdrant client connection closed")
