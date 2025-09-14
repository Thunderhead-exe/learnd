"""
Tests for the core Learnd functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from learnd.models import Concept, LearndConfig
from learnd.core import LearndCore


@pytest.fixture
def config():
    """Test configuration."""
    return LearndConfig(
        primary_layer_capacity=100,
        promotion_threshold=5,
        demotion_threshold=1,
        openai_api_key="test_key"
    )


@pytest.fixture
def mock_core(config):
    """Mock core with dependencies."""
    core = LearndCore(config)
    core.db_manager = AsyncMock()
    core.embedding_manager = AsyncMock()
    core.concept_extractor = AsyncMock()
    core.clustering_manager = AsyncMock()
    return core


@pytest.mark.asyncio
async def test_process_user_input(mock_core):
    """Test processing user input."""
    # Mock responses
    mock_core.concept_extractor.extract_and_validate.return_value = ["machine learning", "neural networks"]
    mock_core.embedding_manager.embed_batch.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_core.db_manager.get_concept.return_value = None  # No existing concept
    mock_core.db_manager.store_concept.return_value = True
    
    # Test
    result = await mock_core.process_user_input("I want to learn about machine learning")
    
    # Assertions
    assert len(result) == 2
    mock_core.concept_extractor.extract_and_validate.assert_called_once()
    mock_core.embedding_manager.embed_batch.assert_called_once()


@pytest.mark.asyncio
async def test_store_concept(mock_core):
    """Test storing a concept."""
    concept = Concept(
        text="test concept",
        embedding=[0.1, 0.2, 0.3],
        frequency=1
    )
    
    mock_core.db_manager.store_concept.return_value = True
    
    result = await mock_core.store_concept(concept)
    
    assert result is True
    mock_core.db_manager.store_concept.assert_called_once_with(concept)


@pytest.mark.asyncio
async def test_retrieve_relevant_concepts(mock_core):
    """Test retrieving relevant concepts."""
    mock_concept = Concept(
        text="test concept",
        embedding=[0.1, 0.2, 0.3],
        frequency=5
    )
    
    mock_core.embedding_manager.embed_text.return_value = [0.1, 0.2, 0.3]
    mock_core.db_manager.search_similar_concepts.return_value = [(mock_concept, 0.9)]
    
    result = await mock_core.retrieve_relevant_concepts("test query")
    
    assert len(result) == 1
    assert result[0].text == "test concept"


@pytest.mark.asyncio
async def test_layer_promotion(mock_core):
    """Test concept promotion to primary layer."""
    mock_core.db_manager.get_layer_statistics.return_value = {"primary_layer_size": 50}
    mock_core.db_manager.move_concept_between_layers.return_value = True
    
    result = await mock_core.promote_to_primary("test_concept_id")
    
    assert result is True
    mock_core.db_manager.move_concept_between_layers.assert_called_once_with("test_concept_id", 1)


@pytest.mark.asyncio
async def test_frequency_update_triggers_promotion(mock_core):
    """Test that frequency updates can trigger layer promotion."""
    mock_concept = Concept(
        id="test_id",
        text="test concept",
        frequency=4,  # Just below promotion threshold
        layer=2
    )
    
    mock_core.db_manager.get_concept.return_value = mock_concept
    mock_core.db_manager.update_concept_frequency.return_value = True
    mock_core.db_manager.get_layer_statistics.return_value = {"primary_layer_size": 50}
    mock_core.db_manager.move_concept_between_layers.return_value = True
    
    # This should trigger promotion (frequency becomes 5, which meets threshold)
    await mock_core.update_concept_frequency("test_id", 1)
    
    # Verify promotion was triggered
    mock_core.db_manager.move_concept_between_layers.assert_called_once_with("test_id", 1)
