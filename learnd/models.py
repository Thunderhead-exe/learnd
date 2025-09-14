"""
Core data models for the Learnd adaptive learning system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import uuid


@dataclass
class Concept:
    """Represents a learned concept with embeddings and metadata."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    embedding: List[float] = field(default_factory=list)
    frequency: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    layer: int = 2  # Default to secondary layer
    cluster_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert concept to dictionary for storage."""
        return {
            "id": self.id,
            "text": self.text,
            "embedding": self.embedding,
            "frequency": self.frequency,
            "last_accessed": self.last_accessed.isoformat(),
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "layer": self.layer,
            "cluster_id": self.cluster_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Concept":
        """Create concept from dictionary."""
        return cls(
            id=data["id"],
            text=data["text"],
            embedding=data["embedding"],
            frequency=data["frequency"],
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data["metadata"],
            layer=data["layer"],
            cluster_id=data.get("cluster_id")
        )


@dataclass
class Cluster:
    """Represents a cluster of related concepts in the secondary layer."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    representative_vector: List[float] = field(default_factory=list)
    member_concepts: List[str] = field(default_factory=list)
    frequency_sum: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    cohesion_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cluster to dictionary for storage."""
        return {
            "id": self.id,
            "representative_vector": self.representative_vector,
            "member_concepts": self.member_concepts,
            "frequency_sum": self.frequency_sum,
            "last_updated": self.last_updated.isoformat(),
            "cohesion_score": self.cohesion_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Cluster":
        """Create cluster from dictionary."""
        return cls(
            id=data["id"],
            representative_vector=data["representative_vector"],
            member_concepts=data["member_concepts"],
            frequency_sum=data["frequency_sum"],
            last_updated=datetime.fromisoformat(data["last_updated"]),
            cohesion_score=data["cohesion_score"]
        )


@dataclass
class FrequencyTracker:
    """Tracks concept access frequencies and manages decay."""
    
    concept_frequencies: Dict[str, int] = field(default_factory=dict)
    access_history: List[tuple] = field(default_factory=list)
    decay_factor: float = 0.95
    promotion_threshold: int = 10
    demotion_threshold: int = 2
    
    def update_frequency(self, concept_id: str, increment: int = 1) -> None:
        """Update frequency for a concept."""
        current_freq = self.concept_frequencies.get(concept_id, 0)
        self.concept_frequencies[concept_id] = current_freq + increment
        self.access_history.append((concept_id, datetime.now()))
    
    def apply_decay(self) -> None:
        """Apply time-based decay to all frequencies."""
        for concept_id in self.concept_frequencies:
            self.concept_frequencies[concept_id] = int(
                self.concept_frequencies[concept_id] * self.decay_factor
            )
    
    def should_promote(self, concept_id: str) -> bool:
        """Check if concept should be promoted to primary layer."""
        return self.concept_frequencies.get(concept_id, 0) >= self.promotion_threshold
    
    def should_demote(self, concept_id: str) -> bool:
        """Check if concept should be demoted to secondary layer."""
        return self.concept_frequencies.get(concept_id, 0) <= self.demotion_threshold


@dataclass
class LearndConfig:
    """Configuration parameters for the Learnd system."""
    
    # Layer Management
    primary_layer_capacity: int = 1000
    promotion_threshold: int = 10
    demotion_threshold: int = 2
    frequency_decay_rate: float = 0.95
    
    # Clustering
    cluster_similarity_threshold: float = 0.8
    max_cluster_size: int = 50
    min_cluster_cohesion: float = 0.6
    clustering_method: str = "kmeans"  # kmeans, hierarchical, dbscan
    
    # Retrieval
    default_similarity_threshold: float = 0.7
    max_retrieval_results: int = 10
    context_window_size: int = 2048
    
    # Maintenance
    cleanup_age_days: int = 30
    rebalance_frequency_hours: int = 24
    optimization_interval_hours: int = 168  # Weekly
    
    # External Services
    mistral_api_key: Optional[str] = None
    qdrant_url: str = "https://your-cluster.gcp.cloud.qdrant.io:6333"
    qdrant_api_key: Optional[str] = None
    collection_name: str = "learnd-concepts"
    
    # Embedding Model
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Secondary LLM (Mistral)
    concept_extraction_model: str = "mistral-large-latest"
    max_concepts_per_input: int = 10


# Pydantic models for MCP request/response validation
class ConceptRequest(BaseModel):
    """Request model for concept operations."""
    text: str = Field(..., description="Text to extract concepts from")
    context: Optional[str] = Field(None, description="Optional context for extraction")


class ConceptResponse(BaseModel):
    """Response model for concept operations."""
    concepts: List[str] = Field(..., description="Extracted concepts")
    concept_ids: List[str] = Field(..., description="Generated concept IDs")


class RetrievalRequest(BaseModel):
    """Request model for concept retrieval."""
    query: str = Field(..., description="Query text for retrieval")
    max_concepts: int = Field(10, description="Maximum concepts to return")
    similarity_threshold: float = Field(0.7, description="Minimum similarity score")


class RetrievalResponse(BaseModel):
    """Response model for concept retrieval."""
    concepts: List[Dict[str, Any]] = Field(..., description="Retrieved concepts")
    context: str = Field(..., description="Formatted context for LLM")


class LayerStatsResponse(BaseModel):
    """Response model for layer statistics."""
    primary_layer_size: int = Field(..., description="Number of concepts in primary layer")
    secondary_layer_size: int = Field(..., description="Number of concepts in secondary layer")
    cluster_count: int = Field(..., description="Number of clusters")
    total_concepts: int = Field(..., description="Total number of concepts")
