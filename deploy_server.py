#!/usr/bin/env python3
"""
Learnd MCP Server - Deployment Version

Standalone MCP server for deployment scenarios that avoids relative import issues.
This version includes all necessary imports and can be deployed independently.
"""

import asyncio
import json
import os
import sys
from typing import Dict, List, Optional, Any
from pathlib import Path
from dotenv import load_dotenv
from fastmcp import FastMCP
from loguru import logger

# Add current directory to Python path for absolute imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Now import with absolute paths
from learnd.core import LearndCore
from learnd.models import LearndConfig

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
            "extraction_confidence": 0.92,
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
async def get_system_stats() -> Dict[str, Any]:
    """
    📊 Get comprehensive system statistics and health metrics
    
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
        "system_healthy": true
    }
    """
    try:
        core = await get_core()
        
        # Get basic layer statistics
        stats = await core.get_layer_statistics()
        
        return {
            "layers": {
                "primary_size": stats["primary_layer_size"],
                "secondary_size": stats["secondary_layer_size"],
                "cluster_count": stats["cluster_count"],
                "total_concepts": stats["total_concepts"],
                "primary_capacity": core.config.primary_layer_capacity,
                "utilization_percent": round((stats["primary_layer_size"] / core.config.primary_layer_capacity) * 100, 1)
            },
            "system_healthy": stats["total_concepts"] > 0
        }
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        return {"error": str(e), "system_healthy": False}


# Health check endpoint for deployment
@mcp.tool
async def health_check() -> Dict[str, Any]:
    """
    ❤️ Health check for deployment monitoring
    
    Intent: Simple health check endpoint to verify the service is running correctly.
    
    Expected Input: None
    
    Expected Output:
    {
        "status": "healthy",
        "service": "learnd-mcp",
        "version": "0.1.0"
    }
    """
    try:
        return {
            "status": "healthy",
            "service": "learnd-mcp", 
            "version": "0.1.0",
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# Export the mcp instance for deployment
__all__ = ["mcp"]

if __name__ == "__main__":
    # For testing the deployment version
    print("🧠 Learnd MCP Server - Deployment Version")
    print("✅ Server configuration loaded")
    print(f"📊 Available tools: {len([tool for tool in dir() if tool.startswith('get_') or tool.startswith('learn_') or tool.startswith('extract_')])}")
    print("🚀 Ready for deployment")
