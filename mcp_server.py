#!/usr/bin/env python3
"""
Learnd MCP Server - Simplified Single-File Implementation

A streamlined Model Context Protocol server for adaptive continuous learning.
This single file contains all functionality needed for deployment.

🧠 Core Concept: Learn from every interaction and provide relevant context
🔄 Auto-Learning: Automatically extracts and stores concepts from user inputs
🎯 Smart Retrieval: Finds relevant learned concepts to enhance LLM responses
"""

import asyncio
import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import tempfile
import warnings

# Environment setup for deployment
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_CACHE', tempfile.gettempdir())
warnings.filterwarnings("ignore")

from fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize MCP server
mcp = FastMCP("learnd")

# ============================================================================
# SIMPLE IN-MEMORY STORAGE (for demonstration - replace with persistent DB)
# ============================================================================

# Simple storage dictionaries
concepts_storage = {}  # concept_id -> {text, frequency, last_used, embedding}
interactions_log = []  # List of all interactions for learning

# ============================================================================
# CORE LEARNING FUNCTIONS
# ============================================================================

def extract_concepts_simple(text: str) -> List[str]:
    """
    Simple concept extraction using basic NLP techniques.
    Extracts meaningful terms, phrases, and topics from text.
    
    Why: We need to identify what the user is talking about to learn from it.
    How: Uses regex patterns and keyword detection to find important concepts.
    """
    if not text or len(text.strip()) < 10:
        return []
    
    # Clean and normalize text
    text = text.lower().strip()
    
    # Extract potential concepts using multiple patterns
    concepts = []
    
    # 1. Technical terms (words with specific patterns)
    tech_patterns = [
        r'\b(?:api|sdk|database|server|client|framework|library|algorithm)\b',
        r'\b(?:machine learning|artificial intelligence|neural network|deep learning)\b',
        r'\b(?:python|javascript|typescript|react|node|docker|kubernetes)\b',
        r'\b(?:authentication|authorization|security|encryption|token)\b',
        r'\b(?:frontend|backend|fullstack|microservice|deployment)\b',
    ]
    
    for pattern in tech_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        concepts.extend(matches)
    
    # 2. Domain-specific terms (capitalized words, acronyms)
    domain_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    concepts.extend([term.lower() for term in domain_terms if len(term) > 3])
    
    # 3. Important phrases (quoted text, "how to" patterns)
    quoted_text = re.findall(r'"([^"]+)"', text)
    concepts.extend(quoted_text)
    
    how_to_patterns = re.findall(r'how to\s+([^.!?]+)', text, re.IGNORECASE)
    concepts.extend([f"how to {pattern.strip()}" for pattern in how_to_patterns])
    
    # 4. Key nouns and noun phrases (simple extraction)
    important_words = re.findall(r'\b(?:error|problem|issue|solution|method|function|class|component|service|system|process|workflow|strategy|approach|technique|implementation|configuration|setup|integration|optimization|performance|scalability|monitoring|testing|debugging|troubleshooting)\b', text, re.IGNORECASE)
    concepts.extend(important_words)
    
    # Clean and deduplicate
    cleaned_concepts = []
    for concept in concepts:
        concept = concept.strip().lower()
        if len(concept) >= 3 and concept not in cleaned_concepts:
            cleaned_concepts.append(concept)
    
    return cleaned_concepts[:10]  # Limit to top 10 concepts

def simple_embedding(text: str) -> List[float]:
    """
    Simple text embedding using character and word features.
    Creates a basic vector representation for similarity matching.
    
    Why: We need to compare concepts for similarity and relevance.
    How: Uses character frequencies and simple word features to create vectors.
    """
    # Normalize text
    text = text.lower().strip()
    
    # Create a simple 100-dimensional embedding
    embedding = [0.0] * 100
    
    # Character frequency features (first 26 dimensions)
    for i, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
        if i < 26:
            embedding[i] = text.count(char) / max(len(text), 1)
    
    # Length features
    embedding[26] = min(len(text) / 50.0, 1.0)  # Normalized length
    embedding[27] = len(text.split()) / max(len(text.split()), 1)  # Words per char
    
    # Simple word presence features (28-70)
    common_words = ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'man', 'car', 'run', 'use', 'big', 'end', 'far', 'off', 'own', 'say']
    for i, word in enumerate(common_words):
        if i < 42:
            embedding[28 + i] = 1.0 if word in text else 0.0
    
    # Technical term presence (70-90)
    tech_terms = ['api', 'code', 'data', 'file', 'user', 'system', 'server', 'client', 'error', 'function', 'class', 'method', 'object', 'array', 'string', 'number', 'boolean', 'null', 'undefined', 'variable']
    for i, term in enumerate(tech_terms):
        if i < 20:
            embedding[70 + i] = 1.0 if term in text else 0.0
    
    # Hash features for remaining dimensions
    text_hash = hash(text)
    for i in range(90, 100):
        embedding[i] = ((text_hash >> i) & 1) * 0.1
    
    return embedding

def calculate_similarity(emb1: List[float], emb2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings."""
    if len(emb1) != len(emb2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(emb1, emb2))
    norm1 = sum(a * a for a in emb1) ** 0.5
    norm2 = sum(b * b for b in emb2) ** 0.5
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def store_concept(concept_text: str) -> str:
    """
    Store a concept in memory with frequency tracking.
    
    Why: We need to remember what the user talks about and how often.
    How: Creates or updates concept entries with usage frequency.
    """
    concept_id = str(uuid.uuid4())
    embedding = simple_embedding(concept_text)
    
    # Check if concept already exists (similar concept)
    for existing_id, existing_concept in concepts_storage.items():
        similarity = calculate_similarity(embedding, existing_concept['embedding'])
        if similarity > 0.8:  # High similarity threshold
            # Update existing concept
            existing_concept['frequency'] += 1
            existing_concept['last_used'] = datetime.now(timezone.utc).isoformat()
            return existing_id
    
    # Store new concept
    concepts_storage[concept_id] = {
        'text': concept_text,
        'frequency': 1,
        'last_used': datetime.now(timezone.utc).isoformat(),
        'embedding': embedding,
        'created': datetime.now(timezone.utc).isoformat()
    }
    
    return concept_id

def find_relevant_concepts(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Find concepts most relevant to a query.
    
    Why: When the user asks something, we want to provide related context from past learning.
    How: Compares query embedding with stored concept embeddings using similarity.
    """
    if not concepts_storage:
        return []
    
    query_embedding = simple_embedding(query)
    concept_scores = []
    
    for concept_id, concept_data in concepts_storage.items():
        similarity = calculate_similarity(query_embedding, concept_data['embedding'])
        # Boost score by frequency (more frequently mentioned = more important)
        boosted_score = similarity * (1 + concept_data['frequency'] * 0.1)
        
        concept_scores.append({
            'id': concept_id,
            'text': concept_data['text'],
            'frequency': concept_data['frequency'],
            'similarity': similarity,
            'score': boosted_score,
            'last_used': concept_data['last_used']
        })
    
    # Sort by boosted score and return top results
    concept_scores.sort(key=lambda x: x['score'], reverse=True)
    return concept_scores[:max_results]

# ============================================================================
# MCP TOOLS - SIMPLE AND EFFECTIVE
# ============================================================================

@mcp.tool
async def learn_from_text(text: str, context: Optional[str] = None) -> Dict[str, Any]:
    """
    🧠 Automatically learn concepts from any text input
    
    WHAT IT DOES:
    - Takes any text and extracts important concepts/topics
    - Stores these concepts with frequency tracking  
    - Updates knowledge base for future use
    
    WHEN TO USE:
    - Call this with every user input to build knowledge
    - Use for documents, conversations, any learning material
    
    INPUT: 
    - text (required): Any text to learn from
    - context (optional): Additional context about the text
    
    OUTPUT:
    {
        "success": true,
        "concepts_learned": ["machine learning", "neural networks"],
        "total_concepts": 2,
        "message": "Successfully learned 2 new concepts"
    }
    
    EXAMPLE:
    learn_from_text("I'm building a React app with authentication")
    -> Learns: "react", "authentication", "app building"
    """
    try:
        # Log the interaction
        interactions_log.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'text': text,
            'context': context,
            'type': 'learning'
        })
        
        # Extract concepts
        concepts = extract_concepts_simple(text)
        
        if not concepts:
            return {
                "success": True,
                "concepts_learned": [],
                "total_concepts": 0,
                "message": "No significant concepts found to learn"
            }
        
        # Store each concept
        stored_concepts = []
        for concept in concepts:
            store_concept(concept)
            stored_concepts.append(concept)
        
        return {
            "success": True,
            "concepts_learned": stored_concepts,
            "total_concepts": len(stored_concepts),
            "message": f"Successfully learned {len(stored_concepts)} concepts",
            "knowledge_base_size": len(concepts_storage)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "concepts_learned": [],
            "total_concepts": 0
        }

@mcp.tool
async def get_relevant_context(query: str, max_concepts: int = 5) -> Dict[str, Any]:
    """
    🔍 Get learned knowledge relevant to a query
    
    WHAT IT DOES:
    - Searches all learned concepts for ones relevant to your query
    - Returns the most relevant concepts with context
    - Provides formatted context to enhance LLM responses
    
    WHEN TO USE:
    - Before generating responses to user questions
    - When you need background knowledge on a topic
    - To check what the system has learned about something
    
    INPUT:
    - query (required): What you want to find relevant knowledge about
    - max_concepts (optional): Maximum number of concepts to return (default: 5)
    
    OUTPUT:
    {
        "relevant_concepts": [
            {
                "text": "machine learning",
                "frequency": 5,
                "similarity": 0.85,
                "last_used": "2024-01-15T10:30:00Z"
            }
        ],
        "context_summary": "Based on previous conversations about: machine learning (mentioned 5 times)...",
        "total_found": 1
    }
    
    EXAMPLE:
    get_relevant_context("How do I train a model?")
    -> Returns: concepts about "machine learning", "training", "models"
    """
    try:
        # Log the query
        interactions_log.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'text': query,
            'type': 'retrieval'
        })
        
        # Find relevant concepts
        relevant = find_relevant_concepts(query, max_concepts)
        
        if not relevant:
            return {
                "relevant_concepts": [],
                "context_summary": "No relevant previous knowledge found for this query.",
                "total_found": 0,
                "suggestion": "Try using learn_from_text to build knowledge first"
            }
        
        # Create context summary
        context_parts = []
        for concept in relevant:
            freq_text = f"mentioned {concept['frequency']} times" if concept['frequency'] > 1 else "mentioned once"
            context_parts.append(f"'{concept['text']}' ({freq_text})")
        
        context_summary = f"Based on previous learning about: {', '.join(context_parts[:3])}"
        if len(relevant) > 3:
            context_summary += f" and {len(relevant) - 3} other related topics"
        
        return {
            "relevant_concepts": relevant,
            "context_summary": context_summary,
            "total_found": len(relevant),
            "knowledge_base_size": len(concepts_storage)
        }
        
    except Exception as e:
        return {
            "relevant_concepts": [],
            "context_summary": f"Error retrieving context: {str(e)}",
            "total_found": 0,
            "error": str(e)
        }

@mcp.tool
async def smart_response_with_learning(user_input: str, intended_response: Optional[str] = None) -> Dict[str, Any]:
    """
    🤖 Complete learning + context pipeline for LLM responses
    
    WHAT IT DOES:
    - Automatically learns from user input
    - Finds relevant context from previous learning
    - Provides enhanced context for better LLM responses
    - This is the main function that combines everything
    
    WHEN TO USE:
    - Call this for EVERY user interaction
    - Use as the primary function for chat applications
    - Perfect for enhancing LLM conversations with learned context
    
    INPUT:
    - user_input (required): What the user said/asked
    - intended_response (optional): LLM's planned response (for learning)
    
    OUTPUT:
    {
        "learning_results": {...},        // What was learned from input
        "relevant_context": {...},       // What context was found
        "enhanced_prompt": "...",        // Formatted prompt with context
        "success": true
    }
    
    EXAMPLE WORKFLOW:
    1. User: "How do I deploy a React app?"
    2. This function:
       - Learns: "react", "deployment", "app"
       - Finds context: Previous mentions of "react", "deployment"
       - Returns enhanced prompt with this context
    3. LLM uses enhanced prompt for better response
    """
    try:
        # Step 1: Learn from user input
        learning_results = await learn_from_text(user_input, "user_interaction")
        
        # Step 2: Get relevant context
        context_results = await get_relevant_context(user_input, max_concepts=8)
        
        # Step 3: Learn from LLM response if provided
        response_learning = None
        if intended_response:
            response_learning = await learn_from_text(intended_response, "llm_response")
        
        # Step 4: Create enhanced prompt
        enhanced_prompt = f"User Input: {user_input}\n\n"
        
        if context_results.get('total_found', 0) > 0:
            enhanced_prompt += f"Relevant Context from Previous Learning:\n"
            enhanced_prompt += f"{context_results['context_summary']}\n\n"
            
            # Add specific concepts
            enhanced_prompt += "Key concepts to consider:\n"
            for concept in context_results['relevant_concepts'][:5]:
                enhanced_prompt += f"- {concept['text']} (frequency: {concept['frequency']})\n"
            enhanced_prompt += "\n"
        
        enhanced_prompt += "Please provide a response that takes into account this learned context."
        
        return {
            "success": True,
            "learning_results": learning_results,
            "relevant_context": context_results,
            "response_learning": response_learning,
            "enhanced_prompt": enhanced_prompt,
            "knowledge_base_size": len(concepts_storage),
            "total_interactions": len(interactions_log)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "enhanced_prompt": f"User Input: {user_input}\n\nNote: Learning system encountered an error.",
            "knowledge_base_size": len(concepts_storage)
        }

@mcp.tool
async def get_learning_stats() -> Dict[str, Any]:
    """
    📊 Get statistics about what the system has learned
    
    WHAT IT DOES:
    - Shows total concepts learned
    - Lists most frequent concepts
    - Provides learning activity overview
    - Helps monitor system learning progress
    
    WHEN TO USE:
    - To check if learning is working
    - To see what topics are most discussed
    - For system monitoring and debugging
    
    OUTPUT:
    {
        "total_concepts": 25,
        "total_interactions": 50,
        "top_concepts": [
            {"text": "react", "frequency": 10},
            {"text": "deployment", "frequency": 8}
        ],
        "recent_activity": "Active learning in progress"
    }
    """
    try:
        if not concepts_storage:
            return {
                "total_concepts": 0,
                "total_interactions": len(interactions_log),
                "top_concepts": [],
                "recent_activity": "No learning data yet - try using learn_from_text",
                "status": "empty"
            }
        
        # Get top concepts by frequency
        concepts_by_freq = []
        for concept_id, concept_data in concepts_storage.items():
            concepts_by_freq.append({
                'text': concept_data['text'],
                'frequency': concept_data['frequency'],
                'last_used': concept_data['last_used']
            })
        
        concepts_by_freq.sort(key=lambda x: x['frequency'], reverse=True)
        top_concepts = concepts_by_freq[:10]
        
        # Recent activity
        recent_interactions = len([i for i in interactions_log if i.get('timestamp', '').startswith(datetime.now().strftime('%Y-%m-%d'))])
        
        activity_status = "active" if recent_interactions > 0 else "inactive"
        
        return {
            "total_concepts": len(concepts_storage),
            "total_interactions": len(interactions_log),
            "recent_interactions_today": recent_interactions,
            "top_concepts": top_concepts,
            "recent_activity": f"Learning is {activity_status} - {recent_interactions} interactions today",
            "status": "healthy",
            "average_concept_frequency": sum(c['frequency'] for c in concepts_storage.values()) / len(concepts_storage) if concepts_storage else 0
        }
        
    except Exception as e:
        return {
            "total_concepts": len(concepts_storage),
            "total_interactions": len(interactions_log),
            "error": str(e),
            "status": "error"
        }

@mcp.tool
async def reset_learning() -> Dict[str, Any]:
    """
    🔄 Reset all learned knowledge (use carefully!)
    
    WHAT IT DOES:
    - Clears all stored concepts
    - Resets interaction log
    - Starts fresh learning from scratch
    
    WHEN TO USE:
    - For testing/development
    - When you want to start over
    - To clear incorrect or unwanted learning
    
    OUTPUT:
    {
        "success": true,
        "message": "All learning data cleared",
        "previous_stats": {...}
    }
    """
    global concepts_storage, interactions_log
    
    try:
        # Save previous stats
        previous_stats = {
            "concepts_count": len(concepts_storage),
            "interactions_count": len(interactions_log)
        }
        
        # Clear all data
        concepts_storage.clear()
        interactions_log.clear()
        
        return {
            "success": True,
            "message": "All learning data has been cleared",
            "previous_stats": previous_stats,
            "current_stats": {
                "concepts_count": 0,
                "interactions_count": 0
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to reset learning data"
        }

# ============================================================================
# HEALTH CHECK AND DIAGNOSTICS
# ============================================================================

@mcp.tool
async def health_check() -> Dict[str, Any]:
    """
    ❤️ Check if the learning system is working properly
    
    WHAT IT DOES:
    - Verifies all components are working
    - Tests basic learning functionality
    - Provides system status
    
    OUTPUT:
    {
        "status": "healthy",
        "learning_functional": true,
        "retrieval_functional": true,
        "knowledge_base_size": 10
    }
    """
    try:
        # Test basic functionality
        test_concepts = extract_concepts_simple("test machine learning")
        test_embedding = simple_embedding("test")
        
        # Test storage (without permanent side effects)
        test_similarity = calculate_similarity([1, 0, 0], [1, 0, 0])
        
        return {
            "status": "healthy",
            "service": "learnd-mcp",
            "version": "2.0-simplified",
            "learning_functional": len(test_concepts) > 0,
            "embedding_functional": len(test_embedding) == 100,
            "similarity_functional": test_similarity == 1.0,
            "knowledge_base_size": len(concepts_storage),
            "total_interactions": len(interactions_log),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "service": "learnd-mcp"
        }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🧠 Learnd MCP Server - Simplified Version")
    print("✅ Single-file implementation loaded")
    print("📊 Available tools:")
    print("  - learn_from_text: Learn concepts from any text")
    print("  - get_relevant_context: Find relevant learned knowledge")
    print("  - smart_response_with_learning: Complete learning + context pipeline")
    print("  - get_learning_stats: View learning statistics")
    print("  - reset_learning: Clear all learned data")
    print("  - health_check: Verify system health")
    print("🚀 Ready for deployment!")
