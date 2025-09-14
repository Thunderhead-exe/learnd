#!/usr/bin/env python3
"""
Learnd MCP Server - Simplified AI Learning Server

Following the official Mistral + Qdrant integration pattern for adaptive learning.
This server provides simple tools for learning and retrieving concepts using AI.

Based on: Mistral + Qdrant Cloud tutorial architecture
"""

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from fastmcp import FastMCP

# Load environment variables
load_dotenv()

# Initialize MCP server
mcp = FastMCP("learnd")

# AI Libraries
try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    print("⚠️ MistralAI not available. Install with: uv add mistralai")

try:
    from qdrant_client import QdrantClient, models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("⚠️ Qdrant client not available. Install with: uv add qdrant-client")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Initialize clients
mistral_client = None
qdrant_client = None

if MISTRAL_AVAILABLE:
    api_key = os.getenv('MISTRAL_API_KEY')
    if api_key:
        mistral_client = Mistral(api_key=api_key)
        print("✅ Mistral client initialized")
    else:
        print("⚠️ MISTRAL_API_KEY not found in environment variables")

if QDRANT_AVAILABLE:
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    if qdrant_url and qdrant_api_key:
        try:
            qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            print("✅ Qdrant client initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize Qdrant: {e}")
    else:
        print("⚠️ QDRANT_URL or QDRANT_API_KEY not found")

# Configuration
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'learnd-concepts')
MISTRAL_MODEL = os.getenv('MISTRAL_MODEL', 'mistral-large-latest')
EMBEDDING_DIMENSION = 1024  # Mistral embed model uses 1024 dimensions

# Simple in-memory interaction log
interactions_log = []

# ============================================================================
# QDRANT OPERATIONS (Following tutorial pattern)
# ============================================================================

async def ensure_collection_exists():
    """Ensure the Qdrant collection exists with correct dimensions."""
    if not qdrant_client:
        return False
    
    try:
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if COLLECTION_NAME not in collection_names:
            # Create new collection
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=models.Distance.COSINE
                )
            )
            print(f"✅ Created Qdrant collection: {COLLECTION_NAME} with {EMBEDDING_DIMENSION} dimensions")
        else:
            # Check if existing collection has correct dimensions
            try:
                collection_info = qdrant_client.get_collection(COLLECTION_NAME)
                existing_dim = collection_info.config.params.vectors.size
                
                if existing_dim != EMBEDDING_DIMENSION:
                    print(f"⚠️ Collection dimension mismatch: expected {EMBEDDING_DIMENSION}, got {existing_dim}")
                    print(f"🔄 Recreating collection with correct dimensions...")
                    
                    # Delete and recreate collection
                    qdrant_client.delete_collection(COLLECTION_NAME)
                    qdrant_client.create_collection(
                        collection_name=COLLECTION_NAME,
                        vectors_config=models.VectorParams(
                            size=EMBEDDING_DIMENSION,
                            distance=models.Distance.COSINE
                        )
                    )
                    print(f"✅ Recreated collection with {EMBEDDING_DIMENSION} dimensions")
                else:
                    print(f"✅ Collection exists with correct dimensions: {EMBEDDING_DIMENSION}")
                    
            except Exception as e:
                print(f"⚠️ Could not check collection dimensions: {e}")
        
        return True
    except Exception as e:
        print(f"⚠️ Failed to ensure collection exists: {e}")
        return False

async def store_in_qdrant(text: str, metadata: Dict[str, Any] = None) -> str:
    """Store text in Qdrant using Mistral for embedding generation."""
    if not qdrant_client or not mistral_client:
        return "error: Missing Qdrant or Mistral client"
    
    try:
        await ensure_collection_exists()
        
        # Generate embedding using Mistral
        embeddings_response = mistral_client.embeddings.create(
            model="mistral-embed",
            inputs=[text]
        )
        embedding = embeddings_response.data[0].embedding
        
        # Create point with metadata
        point_id = str(uuid.uuid4())
        payload = {
            "text": text,
            "stored_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {})
        }
        
        # Store in Qdrant
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            ]
        )
        
        return f"Stored successfully with ID: {point_id}"
        
    except Exception as e:
        return f"Error storing in Qdrant: {str(e)}"

async def find_in_qdrant(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Find similar content in Qdrant."""
    if not qdrant_client or not mistral_client:
        return [{"error": "Missing Qdrant or Mistral client"}]
    
    try:
        await ensure_collection_exists()
        
        # Generate embedding for query
        embeddings_response = mistral_client.embeddings.create(
            model="mistral-embed",
            inputs=[query]
        )
        query_embedding = embeddings_response.data[0].embedding
        
        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True,
            score_threshold=0.3
        )
        
        # Format results
        results = []
        for result in search_results:
            results.append({
                "text": result.payload.get("text", ""),
                "score": round(result.score, 3),
                "stored_at": result.payload.get("stored_at", ""),
                "metadata": {k: v for k, v in result.payload.items() 
                           if k not in ["text", "stored_at"]}
            })
        
        return results
        
    except Exception as e:
        return [{"error": f"Error searching Qdrant: {str(e)}"}]

# ============================================================================
# MCP TOOLS (Simplified, following tutorial pattern)
# ============================================================================

@mcp.tool
async def qdrant_store(text: str, context: Optional[str] = None) -> Dict[str, Any]:
    """
    🗃️ Store information in vector memory
    
    WHAT IT DOES:
    Stores text in Qdrant vector database using Mistral embeddings.
    Similar to the tutorial's 'qdrant-store' tool but for learning concepts.
    
    WHEN TO USE:
    - Store important information for later recall
    - Save concepts learned from user interactions
    - Build persistent knowledge base
    
    INPUT:
    - text (required): The information to store
    - context (optional): Additional context about the information
    
    OUTPUT:
    {
        "success": true,
        "message": "Stored successfully with ID: abc-123",
        "stored_text": "The stored information...",
        "timestamp": "2024-01-15T10:30:00Z"
    }
    """
    try:
        # Log the interaction
        interactions_log.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'text': text,
            'context': context,
            'action': 'store'
        })
        
        # Prepare metadata
        metadata = {"interaction_context": context} if context else {}
        
        # Store in Qdrant
        result_message = await store_in_qdrant(text, metadata)
        
        return {
            "success": "error" not in result_message.lower(),
            "message": result_message,
            "stored_text": text[:100] + "..." if len(text) > 100 else text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "collection": COLLECTION_NAME
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to store information"
        }

@mcp.tool
async def qdrant_find(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    🔍 Find similar information from vector memory
    
    WHAT IT DOES:
    Searches Qdrant vector database for information similar to the query.
    Similar to the tutorial's 'qdrant-find' tool but for learned concepts.
    
    WHEN TO USE:
    - Recall previously stored information
    - Find relevant context for current questions
    - Retrieve learned concepts related to a topic
    
    INPUT:
    - query (required): What to search for
    - max_results (optional): Maximum number of results (default: 5)
    
    OUTPUT:
    {
        "results": [
            {
                "text": "Found information...",
                "score": 0.95,
                "stored_at": "2024-01-15T10:30:00Z"
            }
        ],
        "total_found": 1,
        "query": "original search query"
    }
    """
    try:
        # Log the interaction
        interactions_log.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'query': query,
            'action': 'find'
        })
        
        # Search in Qdrant
        results = await find_in_qdrant(query, max_results)
        
        # Filter out error results
        valid_results = [r for r in results if "error" not in r]
        error_results = [r for r in results if "error" in r]
        
        if error_results:
            return {
                "success": False,
                "error": error_results[0]["error"],
                "results": [],
                "total_found": 0,
                "query": query
            }
        
        return {
            "success": True,
            "results": valid_results,
            "total_found": len(valid_results),
            "query": query,
            "collection": COLLECTION_NAME
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "results": [],
            "total_found": 0,
            "query": query
        }

@mcp.tool
async def learn_from_interaction(
    user_input: str, 
    llm_response: Optional[str] = None,
    importance: str = "normal"
) -> Dict[str, Any]:
    """
    🧠 Learn and store concepts from user interaction
    
    WHAT IT DOES:
    Extracts important concepts from user interactions using Mistral AI
    and stores them in Qdrant for future retrieval.
    
    WHEN TO USE:
    - Process every user-LLM interaction
    - Automatically build knowledge from conversations
    - Extract and store key information
    
    INPUT:
    - user_input (required): What the user said/asked
    - llm_response (optional): The LLM's response
    - importance (optional): "low", "normal", "high" - affects storage priority
    
    OUTPUT:
    {
        "success": true,
        "concepts_stored": ["concept1", "concept2"],
        "storage_results": ["Stored successfully...", "..."],
        "total_stored": 2
    }
    """
    try:
        # Use Mistral to extract key concepts
        if not mistral_client:
            return {
                "success": False,
                "error": "Mistral client not available",
                "concepts_stored": [],
                "total_stored": 0
            }
        
        # Create extraction prompt
        extraction_prompt = f"""
Extract the 3-5 most important concepts, topics, or pieces of information from this interaction:

User: {user_input}
{f"Assistant: {llm_response}" if llm_response else ""}

Return only a JSON array of strings with the key concepts:
["concept1", "concept2", "concept3"]
"""
        
        # Get concepts from Mistral
        response = mistral_client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.3,
            max_tokens=200
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse concepts
        try:
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            if start_idx != -1 and end_idx != -1:
                concepts = json.loads(content[start_idx:end_idx])
            else:
                concepts = []
        except:
            concepts = []
        
        if not concepts:
            return {
                "success": True,
                "message": "No significant concepts found to store",
                "concepts_stored": [],
                "total_stored": 0
            }
        
        # Store each concept
        storage_results = []
        for concept in concepts:
            metadata = {
                "importance": importance,
                "source": "interaction",
                "user_input": user_input[:200],  # First 200 chars
                "interaction_id": str(uuid.uuid4())[:8]
            }
            
            result = await store_in_qdrant(concept, metadata)
            storage_results.append(result)
        
        return {
            "success": True,
            "concepts_stored": concepts,
            "storage_results": storage_results,
            "total_stored": len(concepts),
            "interaction_logged": True
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "concepts_stored": [],
            "total_stored": 0
        }

@mcp.tool
async def get_relevant_context(query: str, max_concepts: int = 5) -> Dict[str, Any]:
    """
    🎯 Get relevant learned context for a query
    
    WHAT IT DOES:
    Searches for relevant learned concepts and formats them as context
    to enhance LLM responses.
    
    WHEN TO USE:
    - Before generating responses to user questions
    - Get background knowledge on a topic
    - Enhance LLM responses with learned context
    
    INPUT:
    - query (required): The user's question or topic
    - max_concepts (optional): Maximum concepts to retrieve (default: 5)
    
    OUTPUT:
    {
        "context_found": true,
        "formatted_context": "Based on previous learning: concept1, concept2...",
        "concepts": [...],
        "total_found": 2
    }
    """
    try:
        # Search for relevant concepts
        search_result = await qdrant_find(query, max_concepts)
        
        if not search_result["success"] or not search_result["results"]:
            return {
                "context_found": False,
                "formatted_context": "No relevant previous learning found for this query.",
                "concepts": [],
                "total_found": 0,
                "suggestion": "Try using learn_from_interaction to build knowledge first"
            }
        
        # Format context
        concepts = search_result["results"]
        context_parts = []
        
        for concept in concepts:
            score_text = f"(relevance: {concept['score']})"
            context_parts.append(f"'{concept['text']}' {score_text}")
        
        formatted_context = f"Based on previous learning: {', '.join(context_parts[:3])}"
        if len(concepts) > 3:
            formatted_context += f" and {len(concepts) - 3} other related concepts"
        
        return {
            "context_found": True,
            "formatted_context": formatted_context,
            "concepts": concepts,
            "total_found": len(concepts),
            "query": query
        }
        
    except Exception as e:
        return {
            "context_found": False,
            "formatted_context": f"Error retrieving context: {str(e)}",
            "concepts": [],
            "total_found": 0,
            "error": str(e)
        }

@mcp.tool
async def get_system_status() -> Dict[str, Any]:
    """
    ❤️ Check system status and configuration
    
    WHAT IT DOES:
    Provides information about the system's health, configuration,
    and available capabilities.
    
    OUTPUT:
    {
        "status": "healthy",
        "mistral_available": true,
        "qdrant_available": true,
        "collection": "learnd-concepts",
        "total_interactions": 10
    }
    """
    try:
        # Test Qdrant connection
        qdrant_status = "not_configured"
        collection_info = None
        
        if qdrant_client:
            try:
                collections = qdrant_client.get_collections()
                qdrant_status = "connected"
                
                # Try to get collection info
                if COLLECTION_NAME in [col.name for col in collections.collections]:
                    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
            except Exception as e:
                qdrant_status = f"error: {str(e)}"
        
        return {
            "status": "healthy",
            "service": "learnd-mcp",
            "version": "1.0-simplified",
            "mistral": {
                "available": mistral_client is not None,
                "model": MISTRAL_MODEL if mistral_client else None
            },
            "qdrant": {
                "available": qdrant_client is not None,
                "status": qdrant_status,
                "collection": COLLECTION_NAME,
                "points_count": collection_info.points_count if collection_info else 0
            },
            "interactions": {
                "total_logged": len(interactions_log),
                "recent_today": len([i for i in interactions_log 
                                   if i.get('timestamp', '').startswith(datetime.now().strftime('%Y-%m-%d'))])
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "service": "learnd-mcp"
        }

@mcp.tool
async def fix_collection_dimensions() -> Dict[str, Any]:
    """
    🔧 Fix collection dimension mismatch
    
    WHAT IT DOES:
    Checks if the Qdrant collection has the correct dimensions for Mistral embeddings
    and recreates it if there's a mismatch.
    
    WHEN TO USE:
    - When you get dimension mismatch errors
    - After changing embedding models
    - To ensure collection compatibility
    
    OUTPUT:
    {
        "success": true,
        "action": "recreated",
        "old_dimensions": 384,
        "new_dimensions": 1024,
        "message": "Collection recreated with correct dimensions"
    }
    """
    try:
        if not qdrant_client:
            return {
                "success": False,
                "error": "Qdrant client not available"
            }
        
        # Check current collection
        try:
            collection_info = qdrant_client.get_collection(COLLECTION_NAME)
            existing_dim = collection_info.config.params.vectors.size
            points_count = collection_info.points_count
            
            if existing_dim == EMBEDDING_DIMENSION:
                return {
                    "success": True,
                    "action": "no_change_needed",
                    "dimensions": existing_dim,
                    "points_count": points_count,
                    "message": f"Collection already has correct dimensions: {existing_dim}"
                }
            
            # Recreate collection with correct dimensions
            print(f"🔄 Recreating collection: {existing_dim} → {EMBEDDING_DIMENSION} dimensions")
            qdrant_client.delete_collection(COLLECTION_NAME)
            
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=models.Distance.COSINE
                )
            )
            
            return {
                "success": True,
                "action": "recreated",
                "old_dimensions": existing_dim,
                "new_dimensions": EMBEDDING_DIMENSION,
                "previous_points": points_count,
                "message": f"Collection recreated: {existing_dim} → {EMBEDDING_DIMENSION} dimensions"
            }
            
        except Exception as e:
            # Collection might not exist, create it
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=models.Distance.COSINE
                )
            )
            
            return {
                "success": True,
                "action": "created",
                "dimensions": EMBEDDING_DIMENSION,
                "message": f"Collection created with {EMBEDDING_DIMENSION} dimensions"
            }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to fix collection dimensions"
        }

@mcp.tool
async def clear_memory() -> Dict[str, Any]:
    """
    🗑️ Clear all stored memories (use carefully!)
    
    WHAT IT DOES:
    Deletes all data from the Qdrant collection and clears interaction logs.
    This is a destructive operation that cannot be undone.
    
    OUTPUT:
    {
        "success": true,
        "message": "All memories cleared",
        "previous_count": 150
    }
    """
    try:
        previous_count = 0
        
        # Get current count if possible
        if qdrant_client:
            try:
                collection_info = qdrant_client.get_collection(COLLECTION_NAME)
                previous_count = collection_info.points_count
                
                # Delete and recreate collection
                qdrant_client.delete_collection(COLLECTION_NAME)
                await ensure_collection_exists()
                
            except Exception as e:
                print(f"Warning: Could not clear Qdrant collection: {e}")
        
        # Clear interaction log
        interactions_log.clear()
        
        return {
            "success": True,
            "message": "All memories and interaction logs cleared",
            "previous_count": previous_count,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to clear memories"
        }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🧠 Learnd MCP Server - Simplified AI Learning")
    print("✅ Following Mistral + Qdrant tutorial architecture")
    print("🤖 Components:")
    print(f"  - Mistral AI: {'✅ Ready' if mistral_client else '⚠️ Not configured'}")
    print(f"  - Qdrant Cloud: {'✅ Ready' if qdrant_client else '⚠️ Not configured'}")
    print("📊 Available MCP tools:")
    print("  - qdrant_store: Store information in vector memory")
    print("  - qdrant_find: Find similar information from memory")
    print("  - learn_from_interaction: Extract and store concepts from interactions")
    print("  - get_relevant_context: Get learned context for queries")
    print("  - get_system_status: Check system health")
    print("  - fix_collection_dimensions: Fix dimension mismatch issues")
    print("  - clear_memory: Clear all stored memories")
    print("🚀 Ready for deployment!")