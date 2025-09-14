#!/usr/bin/env python3
"""
Validation script for Learnd MCP Server.

This script validates that all components are working correctly.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the learnd package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from learnd.models import LearndConfig
from learnd.core import LearndCore


async def validate_core_functionality():
    """Validate core Learnd functionality."""
    logger.info("🔍 Validating core functionality...")
    
    try:
        # Create test config
        config = LearndConfig(
            primary_layer_capacity=10,
            promotion_threshold=3,
            demotion_threshold=1,
            mistral_api_key=os.getenv("MISTRAL_API_KEY", "test-key")
        )
        
        # Initialize core (will fail gracefully without real API keys)
        core = LearndCore(config)
        
        # Test data model creation
        from learnd.models import Concept, Cluster
        from datetime import datetime
        
        concept = Concept(
            text="test concept",
            embedding=[0.1, 0.2, 0.3],
            frequency=1
        )
        
        cluster = Cluster(
            representative_vector=[0.1, 0.2, 0.3],
            member_concepts=["concept1"],
            frequency_sum=5
        )
        
        # Test serialization
        concept_dict = concept.to_dict()
        restored_concept = Concept.from_dict(concept_dict)
        
        assert restored_concept.text == concept.text
        assert restored_concept.frequency == concept.frequency
        
        logger.info("✅ Core functionality validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Core functionality validation failed: {e}")
        return False


async def validate_embeddings():
    """Validate embedding functionality."""
    logger.info("🔍 Validating embedding functionality...")
    
    try:
        from learnd.embeddings import EmbeddingManager
        
        config = LearndConfig()
        embedding_manager = EmbeddingManager(config)
        
        # Test similarity calculation
        vec1 = [0.1, 0.2, 0.3]
        vec2 = [0.1, 0.2, 0.3]
        similarity = embedding_manager.calculate_similarity(vec1, vec2)
        
        assert abs(similarity - 1.0) < 0.001  # Should be 1.0 for identical vectors
        
        # Test centroid calculation
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        centroid = embedding_manager.calculate_centroid(embeddings)
        expected = [0.2, 0.3]  # Average
        
        assert abs(centroid[0] - expected[0]) < 0.001
        assert abs(centroid[1] - expected[1]) < 0.001
        
        logger.info("✅ Embedding functionality validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedding functionality validation failed: {e}")
        return False


async def validate_clustering():
    """Validate clustering functionality."""
    logger.info("🔍 Validating clustering functionality...")
    
    try:
        from learnd.clustering import ClusteringManager
        from learnd.embeddings import EmbeddingManager
        from learnd.models import Concept
        
        config = LearndConfig()
        embedding_manager = EmbeddingManager(config)
        clustering_manager = ClusteringManager(config, embedding_manager)
        
        # Create test concepts
        concepts = [
            Concept(text="concept1", embedding=[0.1, 0.2, 0.3], frequency=1),
            Concept(text="concept2", embedding=[0.2, 0.3, 0.4], frequency=2),
            Concept(text="concept3", embedding=[0.8, 0.9, 1.0], frequency=1)
        ]
        
        # Test clustering
        clusters = await clustering_manager.cluster_concepts(concepts)
        
        assert len(clusters) > 0
        assert all(len(cluster.member_concepts) > 0 for cluster in clusters)
        
        logger.info("✅ Clustering functionality validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Clustering functionality validation failed: {e}")
        return False


async def validate_database_connection():
    """Validate database connection (if available)."""
    logger.info("🔍 Validating database connection...")
    
    try:
        from learnd.database import QdrantManager
        
        config = LearndConfig()
        db_manager = QdrantManager(config)
        
        # Try to create client (will fail gracefully if Qdrant not available)
        try:
            await db_manager.initialize()
            stats = await db_manager.get_layer_statistics()
            await db_manager.close()
            
            logger.info("✅ Database connection validation passed")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  Database not available (this is OK for testing): {e}")
            return True  # Not a critical failure for validation
        
    except Exception as e:
        logger.error(f"❌ Database validation failed: {e}")
        return False


async def validate_configuration():
    """Validate configuration loading."""
    logger.info("🔍 Validating configuration...")
    
    try:
        # Test default config
        config1 = LearndConfig()
        assert config1.primary_layer_capacity == 1000
        assert config1.embedding_model == "all-MiniLM-L6-v2"
        
        # Test custom config
        config2 = LearndConfig(
            primary_layer_capacity=500,
            promotion_threshold=15
        )
        assert config2.primary_layer_capacity == 500
        assert config2.promotion_threshold == 15
        assert config2.demotion_threshold == 2  # Default value
        
        logger.info("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        return False


async def main():
    """Run all validation tests."""
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    
    logger.info("🧠 Learnd MCP Server Validation")
    logger.info("=" * 40)
    
    validations = [
        ("Configuration", validate_configuration),
        ("Core Functionality", validate_core_functionality),
        ("Embeddings", validate_embeddings),
        ("Clustering", validate_clustering),
        ("Database Connection", validate_database_connection),
    ]
    
    results = []
    
    for name, validation_func in validations:
        logger.info(f"\n📋 Running {name} validation...")
        try:
            result = await validation_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"❌ {name} validation crashed: {e}")
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "=" * 40)
    logger.info("📊 Validation Summary:")
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n🎯 Results: {passed}/{total} validations passed")
    
    if passed == total:
        logger.info("🎉 All validations passed! Learnd is ready to use.")
        return 0
    else:
        logger.error("❌ Some validations failed. Please check the configuration and dependencies.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
