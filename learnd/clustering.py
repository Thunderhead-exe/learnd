"""
Clustering functionality for managing secondary layer concepts.
"""

import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import uuid
from loguru import logger
import numpy as np

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, clustering functionality will be limited")

try:
    from .models import Concept, Cluster, LearndConfig
    from .embeddings import EmbeddingManager
except ImportError:
    from learnd.models import Concept, Cluster, LearndConfig
    from learnd.embeddings import EmbeddingManager


class ClusteringManager:
    """Manages concept clustering for the secondary layer."""
    
    def __init__(self, config: LearndConfig, embedding_manager: EmbeddingManager):
        self.config = config
        self.embedding_manager = embedding_manager
        self.clusters: Dict[str, Cluster] = {}
        
    async def cluster_concepts(self, concepts: List[Concept]) -> List[Cluster]:
        """Cluster a list of concepts and return cluster objects."""
        try:
            if not SKLEARN_AVAILABLE:
                logger.warning("Clustering not available, creating single cluster")
                return await self._create_single_cluster(concepts)
            
            if len(concepts) < 2:
                return await self._create_single_cluster(concepts)
            
            # Extract embeddings
            embeddings = [concept.embedding for concept in concepts]
            embeddings_array = np.array(embeddings)
            
            # Determine optimal number of clusters
            n_clusters = self._determine_optimal_clusters(embeddings_array)
            
            if self.config.clustering_method == "kmeans":
                clusters = await self._kmeans_clustering(concepts, embeddings_array, n_clusters)
            elif self.config.clustering_method == "dbscan":
                clusters = await self._dbscan_clustering(concepts, embeddings_array)
            else:
                # Default to simple similarity-based clustering
                clusters = await self._similarity_clustering(concepts)
            
            logger.info(f"Created {len(clusters)} clusters from {len(concepts)} concepts")
            return clusters
            
        except Exception as e:
            logger.error(f"Failed to cluster concepts: {e}")
            return await self._create_single_cluster(concepts)
    
    def _determine_optimal_clusters(self, embeddings: np.ndarray) -> int:
        """Determine optimal number of clusters using elbow method."""
        try:
            n_samples = len(embeddings)
            
            # Rule of thumb: sqrt(n/2) clusters, with min/max bounds
            optimal = max(2, min(10, int(np.sqrt(n_samples / 2))))
            
            # Ensure we don't exceed max cluster size constraints
            max_clusters = n_samples // self.config.max_cluster_size + 1
            
            return min(optimal, max_clusters)
            
        except Exception as e:
            logger.error(f"Failed to determine optimal clusters: {e}")
            return 2
    
    async def _kmeans_clustering(
        self, 
        concepts: List[Concept], 
        embeddings: np.ndarray, 
        n_clusters: int
    ) -> List[Cluster]:
        """Perform K-means clustering."""
        try:
            # Normalize embeddings
            scaler = StandardScaler()
            normalized_embeddings = scaler.fit_transform(embeddings)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(normalized_embeddings)
            
            # Calculate silhouette score for quality assessment
            silhouette = silhouette_score(normalized_embeddings, cluster_labels)
            logger.debug(f"K-means silhouette score: {silhouette:.3f}")
            
            # Create cluster objects
            clusters = []
            for cluster_id in range(n_clusters):
                cluster_concepts = [
                    concepts[i] for i, label in enumerate(cluster_labels) 
                    if label == cluster_id
                ]
                
                if cluster_concepts:
                    cluster = await self._create_cluster_from_concepts(cluster_concepts)
                    cluster.cohesion_score = silhouette
                    clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            logger.error(f"K-means clustering failed: {e}")
            return await self._create_single_cluster(concepts)
    
    async def _dbscan_clustering(self, concepts: List[Concept], embeddings: np.ndarray) -> List[Cluster]:
        """Perform DBSCAN clustering."""
        try:
            # Normalize embeddings
            scaler = StandardScaler()
            normalized_embeddings = scaler.fit_transform(embeddings)
            
            # Perform DBSCAN clustering
            eps = 1 - self.config.cluster_similarity_threshold  # Convert similarity to distance
            dbscan = DBSCAN(eps=eps, min_samples=2, metric='euclidean')
            cluster_labels = dbscan.fit_predict(normalized_embeddings)
            
            # Create cluster objects
            clusters = []
            unique_labels = set(cluster_labels)
            
            for label in unique_labels:
                if label == -1:  # Noise points
                    continue
                    
                cluster_concepts = [
                    concepts[i] for i, l in enumerate(cluster_labels) 
                    if l == label
                ]
                
                if cluster_concepts:
                    cluster = await self._create_cluster_from_concepts(cluster_concepts)
                    clusters.append(cluster)
            
            # Handle noise points as individual clusters if needed
            noise_concepts = [
                concepts[i] for i, label in enumerate(cluster_labels) 
                if label == -1
            ]
            
            for concept in noise_concepts:
                cluster = await self._create_cluster_from_concepts([concept])
                clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            logger.error(f"DBSCAN clustering failed: {e}")
            return await self._create_single_cluster(concepts)
    
    async def _similarity_clustering(self, concepts: List[Concept]) -> List[Cluster]:
        """Simple similarity-based clustering."""
        try:
            clusters = []
            unclustered_concepts = concepts.copy()
            
            while unclustered_concepts:
                # Start new cluster with first unclustered concept
                seed_concept = unclustered_concepts.pop(0)
                cluster_concepts = [seed_concept]
                
                # Find similar concepts
                remaining_concepts = unclustered_concepts.copy()
                for concept in remaining_concepts:
                    similarity = self.embedding_manager.calculate_similarity(
                        seed_concept.embedding, concept.embedding
                    )
                    
                    if similarity >= self.config.cluster_similarity_threshold:
                        cluster_concepts.append(concept)
                        unclustered_concepts.remove(concept)
                
                # Create cluster
                cluster = await self._create_cluster_from_concepts(cluster_concepts)
                clusters.append(cluster)
                
                # Prevent clusters from getting too large
                if len(cluster_concepts) >= self.config.max_cluster_size:
                    break
            
            return clusters
            
        except Exception as e:
            logger.error(f"Similarity clustering failed: {e}")
            return await self._create_single_cluster(concepts)
    
    async def _create_cluster_from_concepts(self, concepts: List[Concept]) -> Cluster:
        """Create a cluster object from a list of concepts."""
        try:
            cluster_id = str(uuid.uuid4())
            
            # Calculate representative vector (centroid)
            embeddings = [concept.embedding for concept in concepts]
            representative_vector = self.embedding_manager.calculate_centroid(embeddings)
            
            # Calculate total frequency
            frequency_sum = sum(concept.frequency for concept in concepts)
            
            # Update concept cluster assignments
            concept_ids = []
            for concept in concepts:
                concept.cluster_id = cluster_id
                concept_ids.append(concept.id)
            
            # Calculate cohesion score
            cohesion_score = await self._calculate_cohesion(concepts, representative_vector)
            
            cluster = Cluster(
                id=cluster_id,
                representative_vector=representative_vector,
                member_concepts=concept_ids,
                frequency_sum=frequency_sum,
                last_updated=datetime.now(),
                cohesion_score=cohesion_score
            )
            
            return cluster
            
        except Exception as e:
            logger.error(f"Failed to create cluster from concepts: {e}")
            # Return basic cluster
            return Cluster(
                id=str(uuid.uuid4()),
                representative_vector=[],
                member_concepts=[concept.id for concept in concepts],
                frequency_sum=sum(concept.frequency for concept in concepts),
                last_updated=datetime.now(),
                cohesion_score=0.0
            )
    
    async def _create_single_cluster(self, concepts: List[Concept]) -> List[Cluster]:
        """Create a single cluster containing all concepts."""
        if not concepts:
            return []
        
        cluster = await self._create_cluster_from_concepts(concepts)
        return [cluster]
    
    async def _calculate_cohesion(self, concepts: List[Concept], centroid: List[float]) -> float:
        """Calculate cluster cohesion score."""
        try:
            if len(concepts) <= 1 or not centroid:
                return 1.0
            
            similarities = []
            for concept in concepts:
                similarity = self.embedding_manager.calculate_similarity(
                    concept.embedding, centroid
                )
                similarities.append(similarity)
            
            return float(np.mean(similarities))
            
        except Exception as e:
            logger.error(f"Failed to calculate cohesion: {e}")
            return 0.0
    
    async def assign_concept_to_cluster(
        self, 
        concept: Concept, 
        existing_clusters: List[Cluster]
    ) -> Optional[str]:
        """Assign a concept to the most similar existing cluster."""
        try:
            if not existing_clusters:
                return None
            
            best_cluster = None
            best_similarity = -1.0
            
            for cluster in existing_clusters:
                if not cluster.representative_vector:
                    continue
                    
                similarity = self.embedding_manager.calculate_similarity(
                    concept.embedding, cluster.representative_vector
                )
                
                if similarity > best_similarity and similarity >= self.config.cluster_similarity_threshold:
                    best_similarity = similarity
                    best_cluster = cluster
            
            if best_cluster and len(best_cluster.member_concepts) < self.config.max_cluster_size:
                return best_cluster.id
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to assign concept to cluster: {e}")
            return None
    
    async def merge_clusters(self, cluster1: Cluster, cluster2: Cluster) -> Cluster:
        """Merge two clusters into one."""
        try:
            # Combine member concepts
            all_member_ids = cluster1.member_concepts + cluster2.member_concepts
            
            # Calculate new representative vector
            all_embeddings = []
            if cluster1.representative_vector:
                all_embeddings.append(cluster1.representative_vector)
            if cluster2.representative_vector:
                all_embeddings.append(cluster2.representative_vector)
            
            if all_embeddings:
                new_representative = self.embedding_manager.calculate_centroid(all_embeddings)
            else:
                new_representative = []
            
            # Combine frequencies
            combined_frequency = cluster1.frequency_sum + cluster2.frequency_sum
            
            # Create merged cluster
            merged_cluster = Cluster(
                id=str(uuid.uuid4()),
                representative_vector=new_representative,
                member_concepts=all_member_ids,
                frequency_sum=combined_frequency,
                last_updated=datetime.now(),
                cohesion_score=(cluster1.cohesion_score + cluster2.cohesion_score) / 2
            )
            
            logger.info(f"Merged clusters {cluster1.id} and {cluster2.id} into {merged_cluster.id}")
            return merged_cluster
            
        except Exception as e:
            logger.error(f"Failed to merge clusters: {e}")
            return cluster1  # Return original cluster on error
    
    async def update_cluster_representative(
        self, 
        cluster: Cluster, 
        member_concepts: List[Concept]
    ) -> Cluster:
        """Update the representative vector of a cluster."""
        try:
            if not member_concepts:
                return cluster
            
            # Recalculate representative vector
            embeddings = [concept.embedding for concept in member_concepts]
            new_representative = self.embedding_manager.calculate_centroid(embeddings)
            
            # Update frequency sum
            new_frequency_sum = sum(concept.frequency for concept in member_concepts)
            
            # Recalculate cohesion
            new_cohesion = await self._calculate_cohesion(member_concepts, new_representative)
            
            # Update cluster
            cluster.representative_vector = new_representative
            cluster.frequency_sum = new_frequency_sum
            cluster.cohesion_score = new_cohesion
            cluster.last_updated = datetime.now()
            
            return cluster
            
        except Exception as e:
            logger.error(f"Failed to update cluster representative: {e}")
            return cluster
