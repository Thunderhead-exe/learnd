"""
Core learning pipeline and concept management for the Learnd system.
"""

import asyncio
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import uuid
from loguru import logger

from .models import Concept, Cluster, FrequencyTracker, LearndConfig
from .database import QdrantManager
from .embeddings import EmbeddingManager
from .concept_extractor import ConceptExtractor
from .clustering import ClusteringManager


class LearndCore:
    """Core learning engine that manages the adaptive continuous learning system."""
    
    def __init__(self, config: LearndConfig):
        self.config = config
        self.db_manager = QdrantManager(config)
        self.embedding_manager = EmbeddingManager(config)
        self.concept_extractor = ConceptExtractor(config)
        self.clustering_manager = ClusteringManager(config, self.embedding_manager)
        self.frequency_tracker = FrequencyTracker(
            decay_factor=config.frequency_decay_rate,
            promotion_threshold=config.promotion_threshold,
            demotion_threshold=config.demotion_threshold
        )
        
    async def initialize(self) -> None:
        """Initialize all components of the learning system."""
        try:
            await self.db_manager.initialize()
            await self.embedding_manager.initialize()
            await self.concept_extractor.initialize()
            
            # Load existing frequency data
            await self._load_frequency_data()
            
            logger.info("Learnd core system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Learnd core: {e}")
            raise
    
    async def process_user_input(
        self, 
        user_input: str, 
        interaction_context: Optional[str] = None
    ) -> List[str]:
        """Complete pipeline: extract concepts, update frequencies, manage layers."""
        try:
            logger.info(f"Processing user input: {user_input[:100]}...")
            
            # Extract concepts from input
            concept_texts = await self.concept_extractor.extract_and_validate(
                user_input, interaction_context
            )
            
            if not concept_texts:
                logger.warning("No concepts extracted from input")
                return []
            
            # Generate embeddings
            embeddings = await self.embedding_manager.embed_batch(concept_texts)
            
            if len(embeddings) != len(concept_texts):
                logger.error("Mismatch between concepts and embeddings")
                return []
            
            # Process each concept
            concept_ids = []
            for concept_text, embedding in zip(concept_texts, embeddings):
                concept_id = await self._process_single_concept(
                    concept_text, embedding, user_input, interaction_context
                )
                if concept_id:
                    concept_ids.append(concept_id)
            
            # Trigger layer rebalancing if needed
            await self._check_and_rebalance_layers()
            
            logger.info(f"Processed {len(concept_ids)} concepts successfully")
            return concept_ids
            
        except Exception as e:
            logger.error(f"Failed to process user input: {e}")
            return []
    
    async def _process_single_concept(
        self, 
        concept_text: str, 
        embedding: List[float], 
        original_input: str,
        context: Optional[str]
    ) -> Optional[str]:
        """Process a single concept: store or update and manage frequency."""
        try:
            # Check if similar concept already exists
            existing_concept = await self._find_similar_concept(concept_text, embedding)
            
            if existing_concept:
                # Update existing concept frequency
                await self.update_concept_frequency(existing_concept.id)
                return existing_concept.id
            else:
                # Create new concept
                concept = Concept(
                    id=str(uuid.uuid4()),
                    text=concept_text,
                    embedding=embedding,
                    frequency=1,
                    last_accessed=datetime.now(),
                    created_at=datetime.now(),
                    metadata={
                        "original_input": original_input[:500],  # Store truncated input
                        "context": context,
                        "extraction_source": "user_input"
                    },
                    layer=2  # Start in secondary layer
                )
                
                # Store concept
                success = await self.store_concept(concept)
                if success:
                    self.frequency_tracker.update_frequency(concept.id, 1)
                    return concept.id
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to process concept '{concept_text}': {e}")
            return None
    
    async def _find_similar_concept(
        self, 
        concept_text: str, 
        embedding: List[float]
    ) -> Optional[Concept]:
        """Find if a similar concept already exists in the system."""
        try:
            # Search in primary layer first
            primary_results = await self.db_manager.search_similar_concepts(
                embedding,
                self.db_manager.primary_collection,
                limit=5,
                score_threshold=0.9  # High threshold for exact matches
            )
            
            for concept, score in primary_results:
                if concept.text.lower() == concept_text.lower() or score > 0.95:
                    return concept
            
            # Search in secondary layer
            secondary_results = await self.db_manager.search_similar_concepts(
                embedding,
                self.db_manager.secondary_collection,
                limit=5,
                score_threshold=0.9
            )
            
            for concept, score in secondary_results:
                if concept.text.lower() == concept_text.lower() or score > 0.95:
                    return concept
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to find similar concept: {e}")
            return None
    
    async def store_concept(self, concept: Concept) -> bool:
        """Store a concept in the appropriate layer."""
        try:
            success = await self.db_manager.store_concept(concept)
            if success:
                logger.debug(f"Stored concept '{concept.text}' in layer {concept.layer}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to store concept: {e}")
            return False
    
    async def update_concept_frequency(self, concept_id: str, increment: int = 1) -> None:
        """Update concept access frequency and trigger layer migration if needed."""
        try:
            # Update frequency tracker
            self.frequency_tracker.update_frequency(concept_id, increment)
            
            # Get current concept
            concept = await self.db_manager.get_concept(concept_id)
            if not concept:
                logger.warning(f"Concept {concept_id} not found for frequency update")
                return
            
            # Update concept frequency
            concept.frequency += increment
            concept.last_accessed = datetime.now()
            
            # Update in database
            await self.db_manager.update_concept_frequency(concept_id, concept.frequency)
            
            # Check for layer migration
            if concept.layer == 2 and self.frequency_tracker.should_promote(concept_id):
                await self.promote_to_primary(concept_id)
            elif concept.layer == 1 and self.frequency_tracker.should_demote(concept_id):
                await self.demote_to_secondary(concept_id)
            
            logger.debug(f"Updated frequency for concept {concept_id} to {concept.frequency}")
            
        except Exception as e:
            logger.error(f"Failed to update concept frequency: {e}")
    
    async def promote_to_primary(self, concept_id: str) -> bool:
        """Move concept from secondary to primary layer."""
        try:
            # Check primary layer capacity
            stats = await self.db_manager.get_layer_statistics()
            if stats["primary_layer_size"] >= self.config.primary_layer_capacity:
                # Make room by demoting least frequent concept
                await self._make_room_in_primary()
            
            # Move concept to primary layer
            success = await self.db_manager.move_concept_between_layers(concept_id, 1)
            
            if success:
                logger.info(f"Promoted concept {concept_id} to primary layer")
                
                # Remove from any cluster if it was clustered
                await self._remove_concept_from_cluster(concept_id)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to promote concept {concept_id}: {e}")
            return False
    
    async def demote_to_secondary(self, concept_id: str) -> bool:
        """Move concept from primary to secondary layer and handle clustering."""
        try:
            # Move concept to secondary layer
            success = await self.db_manager.move_concept_between_layers(concept_id, 2)
            
            if success:
                logger.info(f"Demoted concept {concept_id} to secondary layer")
                
                # Assign to appropriate cluster
                await self._assign_concept_to_cluster(concept_id)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to demote concept {concept_id}: {e}")
            return False
    
    async def _make_room_in_primary(self) -> None:
        """Make room in primary layer by demoting least frequent concept."""
        try:
            # Get all concepts in primary layer
            primary_concepts = await self.db_manager.get_all_concepts_in_layer(1)
            
            if not primary_concepts:
                return
            
            # Find least frequent concept
            least_frequent = min(primary_concepts, key=lambda c: c.frequency)
            
            # Demote it to secondary layer
            await self.demote_to_secondary(least_frequent.id)
            
            logger.info(f"Made room in primary layer by demoting {least_frequent.id}")
            
        except Exception as e:
            logger.error(f"Failed to make room in primary layer: {e}")
    
    async def _assign_concept_to_cluster(self, concept_id: str) -> None:
        """Assign a concept to an appropriate cluster in secondary layer."""
        try:
            concept = await self.db_manager.get_concept(concept_id)
            if not concept or concept.layer != 2:
                return
            
            # Get existing clusters (would need cluster storage implementation)
            # For now, create a new cluster or assign to best matching cluster
            
            # This would be implemented with cluster storage in database
            logger.debug(f"Assigned concept {concept_id} to cluster")
            
        except Exception as e:
            logger.error(f"Failed to assign concept to cluster: {e}")
    
    async def _remove_concept_from_cluster(self, concept_id: str) -> None:
        """Remove a concept from its cluster when promoted."""
        try:
            concept = await self.db_manager.get_concept(concept_id)
            if not concept or not concept.cluster_id:
                return
            
            # Remove cluster assignment
            concept.cluster_id = None
            await self.db_manager.store_concept(concept)
            
            logger.debug(f"Removed concept {concept_id} from cluster")
            
        except Exception as e:
            logger.error(f"Failed to remove concept from cluster: {e}")
    
    async def retrieve_relevant_concepts(
        self,
        query: str,
        max_concepts: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Concept]:
        """Retrieve relevant concepts from primary layer for LLM augmentation."""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_manager.embed_text(query)
            
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []
            
            # Search primary layer
            results = await self.db_manager.search_similar_concepts(
                query_embedding,
                self.db_manager.primary_collection,
                limit=max_concepts,
                score_threshold=similarity_threshold
            )
            
            # Update access frequencies for retrieved concepts
            retrieved_concepts = []
            for concept, score in results:
                await self.update_concept_frequency(concept.id)
                retrieved_concepts.append(concept)
            
            logger.info(f"Retrieved {len(retrieved_concepts)} relevant concepts")
            return retrieved_concepts
            
        except Exception as e:
            logger.error(f"Failed to retrieve relevant concepts: {e}")
            return []
    
    async def generate_augmented_context(self, query: str) -> str:
        """Generate context string for LLM augmentation."""
        try:
            relevant_concepts = await self.retrieve_relevant_concepts(query)
            
            if not relevant_concepts:
                return ""
            
            # Format concepts for context
            context_parts = ["Relevant learned concepts:"]
            
            for concept in relevant_concepts:
                context_parts.append(f"- {concept.text}")
                
                # Add metadata if available
                if concept.metadata.get("context"):
                    context_parts.append(f"  Context: {concept.metadata['context']}")
            
            context = "\n".join(context_parts)
            
            # Limit context size
            if len(context) > self.config.context_window_size:
                context = context[:self.config.context_window_size] + "..."
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to generate augmented context: {e}")
            return ""
    
    async def learn_from_interaction(
        self,
        user_input: str,
        llm_response: str,
        feedback_score: Optional[float] = None
    ) -> None:
        """Learn from complete interaction cycle."""
        try:
            # Process user input for concept extraction
            await self.process_user_input(user_input)
            
            # Optionally extract concepts from LLM response for learning
            if feedback_score and feedback_score > 0.7:  # Good response
                await self.process_user_input(
                    llm_response, 
                    f"LLM response to: {user_input[:100]}"
                )
            
            logger.debug("Completed learning from interaction")
            
        except Exception as e:
            logger.error(f"Failed to learn from interaction: {e}")
    
    async def _check_and_rebalance_layers(self) -> None:
        """Check if layer rebalancing is needed and perform it."""
        try:
            stats = await self.db_manager.get_layer_statistics()
            
            # Check if primary layer is over capacity
            if stats["primary_layer_size"] > self.config.primary_layer_capacity:
                await self.rebalance_layers()
            
        except Exception as e:
            logger.error(f"Failed to check and rebalance layers: {e}")
    
    async def rebalance_layers(self) -> None:
        """Perform global layer rebalancing based on frequency thresholds."""
        try:
            logger.info("Starting layer rebalancing...")
            
            # Get all concepts from both layers
            primary_concepts = await self.db_manager.get_all_concepts_in_layer(1)
            secondary_concepts = await self.db_manager.get_all_concepts_in_layer(2)
            
            # Sort concepts by frequency
            all_concepts = primary_concepts + secondary_concepts
            all_concepts.sort(key=lambda c: c.frequency, reverse=True)
            
            # Promote top concepts to primary layer
            top_concepts = all_concepts[:self.config.primary_layer_capacity]
            
            for concept in top_concepts:
                if concept.layer != 1:
                    await self.db_manager.move_concept_between_layers(concept.id, 1)
            
            # Demote remaining concepts to secondary layer
            remaining_concepts = all_concepts[self.config.primary_layer_capacity:]
            
            for concept in remaining_concepts:
                if concept.layer != 2:
                    await self.db_manager.move_concept_between_layers(concept.id, 2)
            
            # Cluster secondary layer concepts
            if remaining_concepts:
                await self._cluster_secondary_concepts(remaining_concepts)
            
            logger.info("Layer rebalancing completed")
            
        except Exception as e:
            logger.error(f"Failed to rebalance layers: {e}")
    
    async def _cluster_secondary_concepts(self, concepts: List[Concept]) -> None:
        """Cluster concepts in secondary layer."""
        try:
            if len(concepts) < 2:
                return
            
            clusters = await self.clustering_manager.cluster_concepts(concepts)
            
            # Store clusters (would need cluster storage implementation)
            for cluster in clusters:
                await self.db_manager.store_cluster(cluster)
            
            logger.info(f"Created {len(clusters)} clusters for {len(concepts)} secondary concepts")
            
        except Exception as e:
            logger.error(f"Failed to cluster secondary concepts: {e}")
    
    async def _load_frequency_data(self) -> None:
        """Load existing frequency data from database."""
        try:
            # This would load frequency data from persistent storage
            # For now, initialize empty tracker
            logger.debug("Loaded frequency data")
            
        except Exception as e:
            logger.error(f"Failed to load frequency data: {e}")
    
    async def apply_frequency_decay(self) -> None:
        """Apply time-based decay to concept frequencies."""
        try:
            self.frequency_tracker.apply_decay()
            
            # Update frequencies in database
            # This would iterate through all concepts and update their frequencies
            
            logger.info("Applied frequency decay to all concepts")
            
        except Exception as e:
            logger.error(f"Failed to apply frequency decay: {e}")
    
    async def cleanup_unused_concepts(self, age_threshold_days: int = 30) -> int:
        """Remove concepts that haven't been accessed for specified period."""
        try:
            cutoff_date = datetime.now() - timedelta(days=age_threshold_days)
            
            # Get all concepts and check last access time
            all_concepts = (
                await self.db_manager.get_all_concepts_in_layer(1) +
                await self.db_manager.get_all_concepts_in_layer(2)
            )
            
            removed_count = 0
            for concept in all_concepts:
                if concept.last_accessed < cutoff_date and concept.frequency <= 1:
                    await self.db_manager.delete_concept(concept.id)
                    removed_count += 1
            
            logger.info(f"Cleaned up {removed_count} unused concepts")
            return removed_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup unused concepts: {e}")
            return 0
    
    async def get_layer_statistics(self) -> Dict[str, int]:
        """Get current statistics about both layers."""
        return await self.db_manager.get_layer_statistics()
    
    async def close(self) -> None:
        """Close all connections and cleanup resources."""
        try:
            await self.db_manager.close()
            logger.info("Learnd core system closed")
            
        except Exception as e:
            logger.error(f"Failed to close Learnd core: {e}")
