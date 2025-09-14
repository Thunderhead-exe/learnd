"""
Tests for Learnd data models.
"""

import pytest
from datetime import datetime
from learnd.models import Concept, Cluster, FrequencyTracker, LearndConfig


def test_concept_creation():
    """Test creating a concept."""
    concept = Concept(
        text="machine learning",
        embedding=[0.1, 0.2, 0.3],
        frequency=5
    )
    
    assert concept.text == "machine learning"
    assert concept.embedding == [0.1, 0.2, 0.3]
    assert concept.frequency == 5
    assert concept.layer == 2  # Default to secondary layer
    assert isinstance(concept.created_at, datetime)


def test_concept_serialization():
    """Test concept to/from dict conversion."""
    concept = Concept(
        text="test concept",
        embedding=[0.1, 0.2],
        frequency=3,
        metadata={"source": "test"}
    )
    
    # Convert to dict
    concept_dict = concept.to_dict()
    
    assert concept_dict["text"] == "test concept"
    assert concept_dict["frequency"] == 3
    assert concept_dict["metadata"]["source"] == "test"
    
    # Convert back from dict
    restored_concept = Concept.from_dict(concept_dict)
    
    assert restored_concept.text == concept.text
    assert restored_concept.frequency == concept.frequency
    assert restored_concept.metadata == concept.metadata


def test_cluster_creation():
    """Test creating a cluster."""
    cluster = Cluster(
        representative_vector=[0.1, 0.2, 0.3],
        member_concepts=["concept1", "concept2"],
        frequency_sum=10,
        cohesion_score=0.8
    )
    
    assert cluster.representative_vector == [0.1, 0.2, 0.3]
    assert len(cluster.member_concepts) == 2
    assert cluster.frequency_sum == 10
    assert cluster.cohesion_score == 0.8


def test_frequency_tracker():
    """Test frequency tracker functionality."""
    tracker = FrequencyTracker(
        promotion_threshold=5,
        demotion_threshold=2
    )
    
    # Update frequency
    tracker.update_frequency("concept1", 3)
    tracker.update_frequency("concept1", 2)
    
    assert tracker.concept_frequencies["concept1"] == 5
    assert tracker.should_promote("concept1")
    assert not tracker.should_demote("concept1")
    
    # Test demotion threshold
    tracker.concept_frequencies["concept2"] = 1
    assert tracker.should_demote("concept2")
    assert not tracker.should_promote("concept2")


def test_frequency_decay():
    """Test frequency decay functionality."""
    tracker = FrequencyTracker(decay_factor=0.8)
    
    tracker.concept_frequencies["concept1"] = 10
    tracker.concept_frequencies["concept2"] = 5
    
    tracker.apply_decay()
    
    assert tracker.concept_frequencies["concept1"] == 8  # 10 * 0.8
    assert tracker.concept_frequencies["concept2"] == 4  # 5 * 0.8


def test_config_defaults():
    """Test configuration with default values."""
    config = LearndConfig()
    
    assert config.primary_layer_capacity == 1000
    assert config.promotion_threshold == 10
    assert config.demotion_threshold == 2
    assert config.embedding_model == "all-MiniLM-L6-v2"
    assert config.clustering_method == "kmeans"


def test_config_custom_values():
    """Test configuration with custom values."""
    config = LearndConfig(
        primary_layer_capacity=500,
        promotion_threshold=15,
        embedding_model="custom-model"
    )
    
    assert config.primary_layer_capacity == 500
    assert config.promotion_threshold == 15
    assert config.embedding_model == "custom-model"
    # Default values should still be present
    assert config.demotion_threshold == 2
