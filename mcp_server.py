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

# Clustering Libraries
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    print("⚠️ Scikit-learn not available. Install with: uv add scikit-learn")

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
PRIMARY_COLLECTION = os.getenv('PRIMARY_COLLECTION', 'learnd-concepts-primary')
SECONDARY_COLLECTION = os.getenv('SECONDARY_COLLECTION', 'learnd-concepts-secondary')
REPRESENTATIVES_COLLECTION = os.getenv('REPRESENTATIVES_COLLECTION', 'learnd-representatives')
MISTRAL_MODEL = os.getenv('MISTRAL_MODEL', 'mistral-large-latest')
EMBEDDING_DIMENSION = 1024  # Mistral embed model uses 1024 dimensions

# Hierarchical store parameters
FREQUENCY_THRESHOLD = 3  # Concepts with frequency >= this stay in primary
CLUSTER_SIZE_THRESHOLD = 5  # Minimum concepts needed to form a cluster
MAX_CLUSTERS = 10  # Maximum number of clusters in secondary layer

# Simple in-memory interaction log
interactions_log = []

# ============================================================================
# QDRANT OPERATIONS (Following tutorial pattern)
# ============================================================================

async def ensure_collections_exist():
    """Ensure all Qdrant collections exist with correct dimensions."""
    if not qdrant_client:
        return False
    
    collections_to_create = [
        PRIMARY_COLLECTION,
        SECONDARY_COLLECTION, 
        REPRESENTATIVES_COLLECTION
    ]
    
    try:
        existing_collections = qdrant_client.get_collections()
        existing_names = [col.name for col in existing_collections.collections]
        
        for collection_name in collections_to_create:
            if collection_name not in existing_names:
                # Create new collection
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=EMBEDDING_DIMENSION,
                        distance=models.Distance.COSINE
                    )
                )
                print(f"✅ Created collection: {collection_name}")
            else:
                # Check dimensions
                try:
                    collection_info = qdrant_client.get_collection(collection_name)
                    existing_dim = collection_info.config.params.vectors.size
                    
                    if existing_dim != EMBEDDING_DIMENSION:
                        print(f"⚠️ {collection_name} dimension mismatch: {existing_dim} → {EMBEDDING_DIMENSION}")
                        qdrant_client.delete_collection(collection_name)
                        qdrant_client.create_collection(
                            collection_name=collection_name,
                            vectors_config=models.VectorParams(
                                size=EMBEDDING_DIMENSION,
                                distance=models.Distance.COSINE
                            )
                        )
                        print(f"✅ Recreated {collection_name} with {EMBEDDING_DIMENSION} dimensions")
                    else:
                        print(f"✅ {collection_name} ready")
                        
                except Exception as e:
                    print(f"⚠️ Could not check {collection_name}: {e}")
        
        return True
    except Exception as e:
        print(f"⚠️ Failed to ensure collections exist: {e}")
        return False

async def store_in_qdrant(text: str, metadata: Dict[str, Any] = None) -> str:
    """Store text in hierarchical Qdrant using frequency-based placement."""
    if not qdrant_client or not mistral_client:
        return "error: Missing Qdrant or Mistral client"
    
    try:
        await ensure_collections_exist()
        
        # Generate embedding using Mistral
        embeddings_response = mistral_client.embeddings.create(
            model="mistral-embed",
            inputs=[text]
        )
        embedding = embeddings_response.data[0].embedding
        
        # Check if concept already exists
        existing_concept = await find_existing_concept(text)
        
        if existing_concept:
            # Update frequency and potentially move between layers
            return await update_concept_frequency(existing_concept, text, embedding, metadata)
        else:
            # New concept - store in primary layer initially
            return await store_new_concept(text, embedding, metadata)
        
    except Exception as e:
        return f"Error storing in Qdrant: {str(e)}"

async def find_existing_concept(text: str) -> Optional[Dict[str, Any]]:
    """Find if a concept already exists in any collection."""
    try:
        # Search in primary collection
        results = await find_in_collection(PRIMARY_COLLECTION, text, limit=1)
        if results and len(results) > 0:
            return {"collection": PRIMARY_COLLECTION, "concept": results[0]}
        
        # Search in secondary collection
        results = await find_in_collection(SECONDARY_COLLECTION, text, limit=1)
        if results and len(results) > 0:
            return {"collection": SECONDARY_COLLECTION, "concept": results[0]}
            
        return None
    except Exception as e:
        print(f"Error finding existing concept: {e}")
        return None

async def store_new_concept(text: str, embedding: List[float], metadata: Dict[str, Any] = None) -> str:
    """Store a new concept in the primary collection."""
    payload = {
        "text": text,
        "stored_at": datetime.now(timezone.utc).isoformat(),
        "frequency": 1,
        "last_accessed": datetime.now(timezone.utc).isoformat(),
        **(metadata or {})
    }
    
    point_id = str(uuid.uuid4())
    qdrant_client.upsert(
        collection_name=PRIMARY_COLLECTION,
        points=[
            models.PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
        ]
    )
    
    return f"Stored new concept in primary layer with ID: {point_id}"

async def update_concept_frequency(existing: Dict[str, Any], text: str, embedding: List[float], metadata: Dict[str, Any] = None) -> str:
    """Update frequency of existing concept and potentially move between layers."""
    collection = existing["collection"]
    concept = existing["concept"]
    current_frequency = concept.get("metadata", {}).get("frequency", 1)
    new_frequency = current_frequency + 1
    
    # Update payload
    updated_payload = concept.get("metadata", {})
    updated_payload.update({
        "frequency": new_frequency,
        "last_accessed": datetime.now(timezone.utc).isoformat(),
        **(metadata or {})
    })
    
    # Determine target collection based on frequency
    target_collection = PRIMARY_COLLECTION if new_frequency >= FREQUENCY_THRESHOLD else SECONDARY_COLLECTION
    
    # If moving between collections, delete from old and add to new
    if collection != target_collection:
        # Delete from current collection
        qdrant_client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=[concept["id"]])
        )
        
        # Add to target collection
        point_id = str(uuid.uuid4())
        qdrant_client.upsert(
            collection_name=target_collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=updated_payload
                )
            ]
        )
        
        return f"Updated concept frequency to {new_frequency} and moved to {target_collection} layer"
    else:
        # Update in same collection
        qdrant_client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=concept["id"],
                    vector=embedding,
                    payload=updated_payload
                )
            ]
        )
        
        return f"Updated concept frequency to {new_frequency} in {collection} layer"

async def find_in_collection(collection_name: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search for similar content in a specific collection."""
    if not qdrant_client or not mistral_client:
        return [{"error": "Missing Qdrant or Mistral client"}]
    
    try:
        # Generate embedding for query
        embeddings_response = mistral_client.embeddings.create(
            model="mistral-embed",
            inputs=[query]
        )
        query_embedding = embeddings_response.data[0].embedding
        
        # Search in specified collection
        search_results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=limit,
            with_payload=True,
            score_threshold=0.3
        )
        
        # Format results
        results = []
        for result in search_results.points:
            results.append({
                "id": result.id,
                "text": result.payload.get("text", ""),
                "score": round(result.score, 3),
                "stored_at": result.payload.get("stored_at", ""),
                "metadata": {k: v for k, v in result.payload.items() 
                           if k not in ["text", "stored_at"]}
            })
        
        return results
        
    except Exception as e:
        return [{"error": f"Error searching {collection_name}: {str(e)}"}]

async def find_in_qdrant(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Find similar content using hierarchical search (primary + secondary with clustering)."""
    if not qdrant_client or not mistral_client:
        return [{"error": "Missing Qdrant or Mistral client"}]
    
    try:
        await ensure_collections_exist()
        
        all_results = []
        
        # 1. Search in primary collection (frequent concepts)
        primary_results = await find_in_collection(PRIMARY_COLLECTION, query, limit)
        if primary_results and not any("error" in r for r in primary_results):
            for result in primary_results:
                result["layer"] = "primary"
                all_results.append(result)
        
        # 2. Search in representatives collection (cluster representatives)
        representative_results = await find_in_collection(REPRESENTATIVES_COLLECTION, query, limit)
        if representative_results and not any("error" in r for r in representative_results):
            # For each representative that matches, load its cluster from secondary
            for rep_result in representative_results:
                rep_result["layer"] = "representative"
                all_results.append(rep_result)
                
                # Load the cluster from secondary collection
                cluster_id = rep_result.get("metadata", {}).get("cluster_id")
                if cluster_id:
                    cluster_results = await load_cluster_from_secondary(cluster_id, query)
                    all_results.extend(cluster_results)
        
        # 3. If no representatives found, search secondary collection directly
        if not representative_results or all("error" in r for r in representative_results):
            secondary_results = await find_in_collection(SECONDARY_COLLECTION, query, limit)
            if secondary_results and not any("error" in r for r in secondary_results):
                for result in secondary_results:
                    result["layer"] = "secondary"
                    all_results.append(result)
        
        # Sort by score and limit results
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results[:limit]
        
    except Exception as e:
        return [{"error": f"Error in hierarchical search: {str(e)}"}]

async def load_cluster_from_secondary(cluster_id: str, query: str) -> List[Dict[str, Any]]:
    """Load all concepts from a specific cluster in the secondary collection."""
    try:
        # Get all points from secondary collection with the cluster_id
        scroll_results = qdrant_client.scroll(
            collection_name=SECONDARY_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="cluster_id",
                        match=models.MatchValue(value=cluster_id)
                    )
                ]
            ),
            limit=100,  # Reasonable limit for cluster size
            with_payload=True
        )
        
        cluster_results = []
        for point in scroll_results[0]:  # scroll_results is (points, next_page_offset)
            cluster_results.append({
                "id": point.id,
                "text": point.payload.get("text", ""),
                "score": 0.8,  # Default score for cluster members
                "stored_at": point.payload.get("stored_at", ""),
                "layer": "secondary_cluster",
                "cluster_id": cluster_id,
                "metadata": {k: v for k, v in point.payload.items() 
                           if k not in ["text", "stored_at"]}
            })
        
        return cluster_results
        
    except Exception as e:
        print(f"Error loading cluster {cluster_id}: {e}")
        return []

# ============================================================================
# CLUSTERING OPERATIONS
# ============================================================================

async def cluster_secondary_concepts():
    """Cluster concepts in secondary collection and create representative vectors."""
    if not CLUSTERING_AVAILABLE or not qdrant_client:
        return {"success": False, "error": "Clustering not available"}
    
    try:
        # Get all concepts from secondary collection
        scroll_results = qdrant_client.scroll(
            collection_name=SECONDARY_COLLECTION,
            limit=1000,  # Reasonable limit
            with_payload=True,
            with_vectors=True
        )
        
        concepts = scroll_results[0]  # (points, next_page_offset)
        
        if len(concepts) < CLUSTER_SIZE_THRESHOLD:
            return {"success": True, "message": f"Not enough concepts to cluster ({len(concepts)} < {CLUSTER_SIZE_THRESHOLD})"}
        
        # Prepare data for clustering
        embeddings = []
        concept_data = []
        
        for point in concepts:
            embeddings.append(point.vector)
            concept_data.append({
                "id": point.id,
                "text": point.payload.get("text", ""),
                "metadata": point.payload
            })
        
        # Determine optimal number of clusters
        n_clusters = min(MAX_CLUSTERS, max(2, len(concepts) // CLUSTER_SIZE_THRESHOLD))
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Create representative vectors and store clusters
        cluster_representatives = []
        
        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            if len(cluster_indices) == 0:
                continue
            
            # Calculate mean vector for cluster
            cluster_embeddings = [embeddings[i] for i in cluster_indices]
            mean_vector = np.mean(cluster_embeddings, axis=0).tolist()
            
            # Get cluster concepts
            cluster_concepts = [concept_data[i] for i in cluster_indices]
            cluster_texts = [concept["text"] for concept in cluster_concepts]
            
            # Create representative concept
            representative_text = f"Cluster {cluster_id}: {', '.join(cluster_texts[:3])}{'...' if len(cluster_texts) > 3 else ''}"
            
            # Store representative in representatives collection
            rep_id = str(uuid.uuid4())
            qdrant_client.upsert(
                collection_name=REPRESENTATIVES_COLLECTION,
                points=[
                    models.PointStruct(
                        id=rep_id,
                        vector=mean_vector,
                        payload={
                            "text": representative_text,
                            "cluster_id": str(cluster_id),
                            "cluster_size": len(cluster_concepts),
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "concept_texts": cluster_texts
                        }
                    )
                ]
            )
            
            # Update secondary collection concepts with cluster_id
            for concept in cluster_concepts:
                updated_payload = concept["metadata"].copy()
                updated_payload["cluster_id"] = str(cluster_id)
                
                qdrant_client.upsert(
                    collection_name=SECONDARY_COLLECTION,
                    points=[
                        models.PointStruct(
                            id=concept["id"],
                            vector=embeddings[concept_data.index(concept)],
                            payload=updated_payload
                        )
                    ]
                )
            
            cluster_representatives.append({
                "cluster_id": cluster_id,
                "size": len(cluster_concepts),
                "representative_text": representative_text
            })
        
        return {
            "success": True,
            "clusters_created": len(cluster_representatives),
            "total_concepts_clustered": len(concepts),
            "representatives": cluster_representatives
        }
        
    except Exception as e:
        return {"success": False, "error": f"Clustering failed: {str(e)}"}

async def promote_concepts_to_primary():
    """Move high-frequency concepts from secondary to primary collection."""
    try:
        # Get all concepts from secondary collection
        scroll_results = qdrant_client.scroll(
            collection_name=SECONDARY_COLLECTION,
            limit=1000,
            with_payload=True,
            with_vectors=True
        )
        
        concepts = scroll_results[0]
        promoted_count = 0
        
        for point in concepts:
            frequency = point.payload.get("frequency", 0)
            
            if frequency >= FREQUENCY_THRESHOLD:
                # Move to primary collection
                point_id = str(uuid.uuid4())
                qdrant_client.upsert(
                    collection_name=PRIMARY_COLLECTION,
                    points=[
                        models.PointStruct(
                            id=point_id,
                            vector=point.vector,
                            payload=point.payload
                        )
                    ]
                )
                
                # Remove from secondary
                qdrant_client.delete(
                    collection_name=SECONDARY_COLLECTION,
                    points_selector=models.PointIdsList(points=[point.id])
                )
                
                promoted_count += 1
        
        return {
            "success": True,
            "promoted_count": promoted_count,
            "message": f"Promoted {promoted_count} concepts to primary layer"
        }
        
    except Exception as e:
        return {"success": False, "error": f"Promotion failed: {str(e)}"}

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
        # Search for relevant concepts using the underlying function
        search_results = await find_in_qdrant(query, max_concepts)
        
        # Check if we got an error
        if search_results and isinstance(search_results, list) and len(search_results) > 0 and "error" in search_results[0]:
            return {
                "context_found": False,
                "formatted_context": f"Error retrieving context: {search_results[0]['error']}",
                "concepts": [],
                "total_found": 0,
                "error": search_results[0]['error']
            }
        
        # Filter out any error results and check if we have valid results
        valid_results = [r for r in search_results if "error" not in r]
        
        if not valid_results:
            return {
                "context_found": False,
                "formatted_context": "No relevant previous learning found for this query.",
                "concepts": [],
                "total_found": 0,
                "suggestion": "Try using learn_from_interaction to build knowledge first"
            }
        
        # Format context
        concepts = valid_results
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
async def cluster_secondary_layer() -> Dict[str, Any]:
    """
    🔄 Cluster concepts in secondary layer and create representative vectors
    
    WHAT IT DOES:
    Groups less frequent concepts in the secondary collection into clusters
    and creates representative vectors for efficient retrieval.
    
    WHEN TO USE:
    - When secondary layer has accumulated enough concepts
    - To optimize search performance for less frequent concepts
    - As part of maintenance routine
    
    OUTPUT:
    {
        "success": true,
        "clusters_created": 3,
        "total_concepts_clustered": 15,
        "representatives": [...]
    }
    """
    return await cluster_secondary_concepts()

@mcp.tool
async def promote_frequent_concepts() -> Dict[str, Any]:
    """
    ⬆️ Promote high-frequency concepts from secondary to primary layer
    
    WHAT IT DOES:
    Moves concepts that have been accessed frequently from the secondary
    layer to the primary layer for faster access.
    
    WHEN TO USE:
    - After clustering to reorganize the hierarchy
    - When concepts become more frequently accessed
    - As part of maintenance routine
    
    OUTPUT:
    {
        "success": true,
        "promoted_count": 5,
        "message": "Promoted 5 concepts to primary layer"
    }
    """
    return await promote_concepts_to_primary()

@mcp.tool
async def get_hierarchy_stats() -> Dict[str, Any]:
    """
    📊 Get statistics about the hierarchical vector store
    
    WHAT IT DOES:
    Provides detailed statistics about the three-layer hierarchy:
    primary concepts, secondary concepts, and cluster representatives.
    
    WHEN TO USE:
    - To monitor the health of the hierarchical system
    - To understand concept distribution across layers
    - For debugging and optimization
    
    OUTPUT:
    {
        "primary_count": 25,
        "secondary_count": 45,
        "representatives_count": 8,
        "clusters": [...],
        "total_concepts": 70
    }
    """
    try:
        if not qdrant_client:
            return {"success": False, "error": "Qdrant client not available"}
        
        # Get counts for each collection
        primary_info = qdrant_client.get_collection(PRIMARY_COLLECTION)
        secondary_info = qdrant_client.get_collection(SECONDARY_COLLECTION)
        representatives_info = qdrant_client.get_collection(REPRESENTATIVES_COLLECTION)
        
        primary_count = primary_info.points_count or 0
        secondary_count = secondary_info.points_count or 0
        representatives_count = representatives_info.points_count or 0
        
        # Get cluster information
        clusters = []
        if representatives_count > 0:
            scroll_results = qdrant_client.scroll(
                collection_name=REPRESENTATIVES_COLLECTION,
                limit=100,
                with_payload=True
            )
            
            for point in scroll_results[0]:
                clusters.append({
                    "cluster_id": point.payload.get("cluster_id"),
                    "size": point.payload.get("cluster_size", 0),
                    "representative_text": point.payload.get("text", "")[:100] + "..."
                })
        
        return {
            "success": True,
            "primary_count": primary_count,
            "secondary_count": secondary_count,
            "representatives_count": representatives_count,
            "clusters": clusters,
            "total_concepts": primary_count + secondary_count,
            "hierarchy_health": "healthy" if representatives_count > 0 else "needs_clustering"
        }
        
    except Exception as e:
        return {"success": False, "error": f"Failed to get hierarchy stats: {str(e)}"}

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
                # Get total count from all collections
                primary_info = qdrant_client.get_collection(PRIMARY_COLLECTION)
                secondary_info = qdrant_client.get_collection(SECONDARY_COLLECTION)
                representatives_info = qdrant_client.get_collection(REPRESENTATIVES_COLLECTION)
                
                previous_count = (primary_info.points_count or 0) + (secondary_info.points_count or 0) + (representatives_info.points_count or 0)
                
                # Delete and recreate all collections
                collections_to_clear = [PRIMARY_COLLECTION, SECONDARY_COLLECTION, REPRESENTATIVES_COLLECTION]
                for collection_name in collections_to_clear:
                    try:
                        qdrant_client.delete_collection(collection_name)
                    except Exception as e:
                        print(f"Warning: Could not clear {collection_name}: {e}")
                
                await ensure_collections_exist()
                
            except Exception as e:
                print(f"Warning: Could not clear Qdrant collections: {e}")
        
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
    print("  - qdrant_store: Store information in hierarchical vector memory")
    print("  - qdrant_find: Find similar information using hierarchical search")
    print("  - learn_from_interaction: Extract and store concepts from interactions")
    print("  - get_relevant_context: Get learned context for queries")
    print("  - get_system_status: Check system health")
    print("  - fix_collection_dimensions: Fix dimension mismatch issues")
    print("  - cluster_secondary_layer: Cluster less frequent concepts")
    print("  - promote_frequent_concepts: Move frequent concepts to primary layer")
    print("  - get_hierarchy_stats: Get hierarchical store statistics")
    print("  - clear_memory: Clear all stored memories")
    print("🚀 Ready for deployment!")