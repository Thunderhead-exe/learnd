"""
Learnd MCP Server - Adaptive Continuous Learning

A Model Context Protocol server that implements adaptive continuous learning 
with hierarchical vector storage using Qdrant Cloud and Mistral AI.

Key Features:
- Automatic concept extraction using Mistral AI
- Hierarchical storage (primary/secondary layers) with frequency-based migration
- Intelligent clustering of infrequent concepts
- Real-time learning from user interactions

Architecture:
- Primary Layer: Fast-access storage for frequently used concepts
- Secondary Layer: Comprehensive storage with clustering for less frequent concepts
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
from dotenv import load_dotenv
from fastmcp import FastMCP
from loguru import logger

from .core import LearndCore
from .models import LearndConfig

# Load environment variables
load_dotenv()

# Initialize MCP server
mcp = FastMCP("learnd")

# Global core instance (initialized when first tool is called)
_core: Optional[LearndCore] = None


async def get_core() -> LearndCore:
    """Get or initialize the Learnd core system."""
    global _core
    if _core is None:
        config = LearndConfig(
            mistral_api_key=os.getenv('MISTRAL_API_KEY'),
            qdrant_url=os.getenv('QDRANT_URL', 'https://your-cluster.gcp.cloud.qdrant.io:6333'),
            qdrant_api_key=os.getenv('QDRANT_API_KEY'),
            collection_name=os.getenv('COLLECTION_NAME', 'learnd-concepts'),
            primary_layer_capacity=int(os.getenv('PRIMARY_LAYER_CAPACITY', 1000)),
            promotion_threshold=int(os.getenv('PROMOTION_THRESHOLD', 10)),
            demotion_threshold=int(os.getenv('DEMOTION_THRESHOLD', 2)),
            embedding_model=os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
            concept_extraction_model=os.getenv('MISTRAL_MODEL', 'mistral-large-latest'),
        )
        _core = LearndCore(config)
        await _core.initialize()
        logger.info("Learnd core system initialized")
    return _core


@mcp.tool
async def learn_from_interaction(
    user_input: str,
    llm_response: Optional[str] = None,
    feedback_score: Optional[float] = None,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    🧠 Learn from a complete user-LLM interaction cycle
    
    This is the primary tool for continuous learning. Use this after each
    user interaction to extract and store concepts automatically.
    
    Intent: Extract concepts from user interactions and learn from them to improve 
    future responses. This builds the knowledge base over time.
    
    Expected Input:
    - user_input (required): The user's input text to learn from
    - llm_response (optional): The LLM's response for reinforcement learning
    - feedback_score (optional): Quality score 0.0-1.0 (higher = better)
    - context (optional): Additional context about the interaction
    
    Expected Output:
    {
        "success": true,
        "concepts_learned": ["machine learning", "artificial intelligence"],
        "concepts_count": 2,
        "learning_summary": "Learned 2 concepts from interaction",
        "feedback_incorporated": true
    }
    """
    try:
        core = await get_core()
        
        # Extract and learn concepts from user input
        concept_ids = await core.process_user_input(user_input, context)
        
        # Optionally learn from LLM response if feedback is positive
        if llm_response and feedback_score and feedback_score > 0.7:
            response_concepts = await core.process_user_input(
                llm_response, 
                f"LLM response to: {user_input[:100]}"
            )
            concept_ids.extend(response_concepts)
        
        # Get concept texts for summary
        concepts_learned = []
        for concept_id in concept_ids:
            concept = await core.db_manager.get_concept(concept_id)
            if concept:
                concepts_learned.append(concept.text)
        
        return {
            "success": True,
            "concepts_learned": concepts_learned,
            "concepts_count": len(concepts_learned),
            "learning_summary": f"Learned {len(concepts_learned)} concepts from interaction",
            "feedback_incorporated": bool(feedback_score and feedback_score > 0.7)
        }
        
    except Exception as e:
        logger.error(f"Failed to learn from interaction: {e}")
        return {
            "success": False, 
            "error": str(e),
            "concepts_learned": [],
            "concepts_count": 0
        }


@mcp.tool
async def get_relevant_context(
    query: str,
    max_concepts: int = 10,
    similarity_threshold: float = 0.7,
    include_metadata: bool = True
) -> Dict[str, Any]:
    """
    🔍 Retrieve relevant learned concepts for LLM context augmentation
    
    Use this before sending queries to your main LLM to provide relevant
    learned context that can improve response quality.
    
    Intent: Find and return concepts learned from previous interactions that are 
    relevant to the current query. This enables the LLM to give more informed responses.
    
    Expected Input:
    - query (required): The query to find relevant concepts for
    - max_concepts (optional): Maximum number of concepts to return (default: 10)
    - similarity_threshold (optional): Minimum similarity score 0.0-1.0 (default: 0.7)
    - include_metadata (optional): Whether to include concept metadata (default: true)
    
    Expected Output:
    {
        "relevant_concepts": [
            {
                "text": "machine learning",
                "frequency": 15,
                "last_accessed": "2024-01-15T10:30:00Z",
                "layer": 1,
                "metadata": {"domain": "AI"}
            }
        ],
        "formatted_context": "Relevant concepts: machine learning (freq: 15)...",
        "total_found": 1
    }
    """
    try:
        core = await get_core()
        
        # Retrieve relevant concepts
        concepts = await core.retrieve_relevant_concepts(
            query, max_concepts, similarity_threshold
        )
        
        # Format response
        relevant_concepts = []
        for concept in concepts:
            concept_data = {
                "text": concept.text,
                "frequency": concept.frequency,
                "last_accessed": concept.last_accessed.isoformat(),
                "layer": concept.layer
            }
            
            if include_metadata:
                concept_data["metadata"] = concept.metadata
            
            relevant_concepts.append(concept_data)
        
        # Generate formatted context for LLM augmentation
        formatted_context = await core.generate_augmented_context(query)
        
        return {
            "relevant_concepts": relevant_concepts,
            "formatted_context": formatted_context,
            "total_found": len(relevant_concepts),
            "query": query
        }
        
    except Exception as e:
        logger.error(f"Failed to get relevant context: {e}")
        return {
            "relevant_concepts": [],
            "formatted_context": "",
            "total_found": 0,
            "error": str(e)
        }


@mcp.tool
async def extract_concepts(
    text: str, 
    context: Optional[str] = None,
    auto_store: bool = True
) -> Dict[str, Any]:
    """
    🎯 Extract concepts from text using Mistral AI
    
    Extracts key concepts, ideas, and entities from input text using
    advanced NLP techniques powered by Mistral AI.
    
    Intent: Analyze text and identify important concepts, terms, and ideas. 
    Useful for processing documents, articles, or any text content.
    
    Expected Input:
    - text (required): Text to extract concepts from
    - context (optional): Additional context to improve extraction quality
    - auto_store (optional): Whether to automatically store concepts (default: true)
    
    Expected Output:
    {
        "concepts": ["machine learning", "neural networks", "AI"],
        "concept_ids": ["uuid1", "uuid2", "uuid3"],
        "extraction_confidence": 0.92,
        "stored_automatically": true,
        "total_extracted": 3
    }
    """
    try:
        core = await get_core()
        
        if auto_store:
            # Use full processing pipeline
            concept_ids = await core.process_user_input(text, context)
            concepts = []
            for concept_id in concept_ids:
                concept = await core.db_manager.get_concept(concept_id)
                if concept:
                    concepts.append(concept.text)
        else:
            # Extract only, don't store
            concepts = await core.concept_extractor.extract_and_validate(text, context)
            concept_ids = []
        
        return {
            "concepts": concepts,
            "concept_ids": concept_ids,
            "extraction_confidence": 0.92,  # Could be calculated from validation
            "stored_automatically": auto_store,
            "total_extracted": len(concepts)
        }
        
    except Exception as e:
        logger.error(f"Failed to extract concepts: {e}")
        return {
            "concepts": [],
            "concept_ids": [],
            "error": str(e),
            "stored_automatically": False
        }


@mcp.tool
async def search_concepts(
    query: str,
    layer: Optional[int] = None,
    limit: int = 20,
    similarity_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    🔎 Search for concepts by similarity across knowledge layers
    
    Find concepts similar to a query across different layers of the
    knowledge base. Useful for exploring existing knowledge.
    
    Intent: Search through stored concepts to find those similar to the query.
    Helps discover related knowledge and explore the knowledge base.
    
    Expected Input:
    - query (required): Search query text
    - layer (optional): Specific layer to search (1=primary, 2=secondary, null=both)
    - limit (optional): Maximum results to return (default: 20)
    - similarity_threshold (optional): Minimum similarity 0.0-1.0 (default: 0.5)
    
    Expected Output:
    {
        "results": [
            {
                "concept": {
                    "text": "machine learning",
                    "frequency": 15,
                    "layer": 1
                },
                "similarity_score": 0.85,
                "rank": 1
            }
        ],
        "total_results": 1,
        "search_layers": [1, 2]
    }
    """
    try:
        core = await get_core()
        
        # Generate query embedding
        query_embedding = await core.embedding_manager.embed_text(query)
        if not query_embedding:
            return {"results": [], "total_results": 0, "error": "Failed to generate query embedding"}
        
        results = []
        search_layers = []
        
        if layer == 1 or layer is None:
            # Search primary layer
            search_layers.append(1)
            primary_results = await core.db_manager.search_similar_concepts(
                query_embedding,
                core.db_manager.primary_collection,
                limit=limit,
                score_threshold=similarity_threshold
            )
            for concept, score in primary_results:
                results.append({
                    "concept": {
                        "id": concept.id,
                        "text": concept.text,
                        "frequency": concept.frequency,
                        "layer": 1,
                        "last_accessed": concept.last_accessed.isoformat()
                    },
                    "similarity_score": score,
                    "layer": 1
                })
        
        if layer == 2 or layer is None:
            # Search secondary layer
            search_layers.append(2)
            secondary_results = await core.db_manager.search_similar_concepts(
                query_embedding,
                core.db_manager.secondary_collection,
                limit=limit,
                score_threshold=similarity_threshold
            )
            for concept, score in secondary_results:
                results.append({
                    "concept": {
                        "id": concept.id,
                        "text": concept.text,
                        "frequency": concept.frequency,
                        "layer": 2,
                        "last_accessed": concept.last_accessed.isoformat()
                    },
                    "similarity_score": score,
                    "layer": 2
                })
        
        # Sort by similarity score and add rank
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        for i, result in enumerate(results[:limit], 1):
            result["rank"] = i
        
        return {
            "results": results[:limit],
            "total_results": len(results),
            "search_layers": search_layers,
            "query": query
        }
        
    except Exception as e:
        logger.error(f"Failed to search concepts: {e}")
        return {"results": [], "total_results": 0, "error": str(e)}


@mcp.tool
async def get_system_stats() -> Dict[str, Any]:
    """
    📊 Get comprehensive system statistics and health metrics
    
    Provides detailed information about the learning system's current state,
    performance metrics, and layer distribution.
    
    Intent: Monitor system health and understand how the knowledge base is growing.
    Essential for maintenance and optimization decisions.
    
    Expected Input: None
    
    Expected Output:
    {
        "layers": {
            "primary_size": 450,
            "secondary_size": 2341,
            "total_concepts": 2791,
            "utilization_percent": 45.0
        },
        "configuration": {
            "promotion_threshold": 10,
            "embedding_model": "all-MiniLM-L6-v2"
        },
        "most_frequent_concept": {
            "text": "machine learning",
            "frequency": 42
        },
        "system_healthy": true
    }
    """
    try:
        core = await get_core()
        
        # Get basic layer statistics
        stats = await core.get_layer_statistics()
        
        # Get most frequent concepts (sample from primary layer)
        primary_concepts = await core.db_manager.get_all_concepts_in_layer(1)
        most_frequent = None
        if primary_concepts:
            most_frequent_concept = max(primary_concepts, key=lambda c: c.frequency)
            most_frequent = {
                "text": most_frequent_concept.text,
                "frequency": most_frequent_concept.frequency
            }
        
        return {
            "layers": {
                "primary_size": stats["primary_layer_size"],
                "secondary_size": stats["secondary_layer_size"],
                "cluster_count": stats["cluster_count"],
                "total_concepts": stats["total_concepts"],
                "primary_capacity": core.config.primary_layer_capacity,
                "utilization_percent": round((stats["primary_layer_size"] / core.config.primary_layer_capacity) * 100, 1)
            },
            "configuration": {
                "promotion_threshold": core.config.promotion_threshold,
                "demotion_threshold": core.config.demotion_threshold,
                "embedding_model": core.config.embedding_model,
                "clustering_method": core.config.clustering_method
            },
            "most_frequent_concept": most_frequent,
            "system_healthy": stats["total_concepts"] > 0
        }
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        return {"error": str(e), "system_healthy": False}


@mcp.tool
async def rebalance_knowledge() -> Dict[str, Any]:
    """
    ⚖️ Rebalance knowledge layers for optimal performance
    
    Reorganizes concepts between layers based on usage frequency to
    maintain optimal retrieval performance.
    
    Intent: Optimize the system by moving frequently used concepts to the primary 
    layer and less frequent ones to secondary layer. Run periodically for best performance.
    
    Expected Input: None
    
    Expected Output:
    {
        "rebalance_completed": true,
        "concepts_promoted": 5,
        "concepts_demoted": 12,
        "stats_before": {"primary_layer_size": 998},
        "stats_after": {"primary_layer_size": 991},
        "primary_utilization": 99.1
    }
    """
    try:
        core = await get_core()
        
        # Get stats before rebalancing
        stats_before = await core.get_layer_statistics()
        
        # Perform rebalancing
        await core.rebalance_layers()
        
        # Get stats after rebalancing
        stats_after = await core.get_layer_statistics()
        
        # Calculate changes
        promoted = max(0, stats_after["primary_layer_size"] - stats_before["primary_layer_size"])
        demoted = max(0, stats_before["primary_layer_size"] - stats_after["primary_layer_size"])
        
        return {
            "rebalance_completed": True,
            "concepts_promoted": promoted,
            "concepts_demoted": demoted,
            "stats_before": stats_before,
            "stats_after": stats_after,
            "primary_utilization": round((stats_after["primary_layer_size"] / core.config.primary_layer_capacity) * 100, 1)
        }
        
    except Exception as e:
        logger.error(f"Failed to rebalance knowledge: {e}")
        return {"rebalance_completed": False, "error": str(e)}


@mcp.tool
async def cleanup_old_concepts(age_threshold_days: int = 30) -> Dict[str, Any]:
    """
    🧹 Clean up old, unused concepts to maintain system efficiency
    
    Removes concepts that haven't been accessed for a specified period
    to keep the knowledge base focused and efficient.
    
    Intent: Remove stale concepts that haven't been used recently to free up 
    storage and improve performance. Part of regular maintenance.
    
    Expected Input:
    - age_threshold_days (optional): Days of inactivity before cleanup (default: 30)
    
    Expected Output:
    {
        "cleanup_completed": true,
        "concepts_removed": 45,
        "age_threshold_days": 30,
        "concepts_remaining": 2746,
        "cleanup_summary": "Removed 45 concepts older than 30 days"
    }
    """
    try:
        core = await get_core()
        
        removed_count = await core.cleanup_unused_concepts(age_threshold_days)
        stats_after = await core.get_layer_statistics()
        
        return {
            "cleanup_completed": True,
            "concepts_removed": removed_count,
            "age_threshold_days": age_threshold_days,
            "concepts_remaining": stats_after["total_concepts"],
            "cleanup_summary": f"Removed {removed_count} concepts older than {age_threshold_days} days"
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup concepts: {e}")
        return {"cleanup_completed": False, "error": str(e)}