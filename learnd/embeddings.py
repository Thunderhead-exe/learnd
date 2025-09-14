"""
Embedding generation and management for the Learnd system.
"""

import asyncio
from typing import List, Optional
from loguru import logger
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available, falling back to OpenAI embeddings")

# Note: We now only use sentence-transformers for embeddings
# This is compatible with Qdrant Cloud and doesn't require external API calls

from .models import LearndConfig


class EmbeddingManager:
    """Manages embedding generation for concepts."""
    
    def __init__(self, config: LearndConfig):
        self.config = config
        self.model: Optional[SentenceTransformer] = None
        
    async def initialize(self) -> None:
        """Initialize the embedding model."""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                # Initialize sentence transformer model
                self.model = SentenceTransformer(self.config.embedding_model)
                logger.info(f"Initialized SentenceTransformer model: {self.config.embedding_model}")
            else:
                raise ValueError("sentence-transformers library not available. Please install it.")
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            if self.model:
                # Use sentence transformer
                embedding = self.model.encode(text, convert_to_tensor=False)
                return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            else:
                raise ValueError("No embedding model available")
                
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            return []
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            if self.model:
                # Use sentence transformer batch processing
                embeddings = self.model.encode(texts, convert_to_tensor=False, batch_size=32)
                return [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings]
            else:
                raise ValueError("No embedding model available")
                
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            return []
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Normalize vectors
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)
            
            # Calculate cosine similarity
            similarity = np.dot(vec1_norm, vec2_norm)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def calculate_centroid(self, embeddings: List[List[float]]) -> List[float]:
        """Calculate the centroid of a list of embeddings."""
        try:
            if not embeddings:
                return []
                
            embeddings_array = np.array(embeddings)
            centroid = np.mean(embeddings_array, axis=0)
            return centroid.tolist()
            
        except Exception as e:
            logger.error(f"Failed to calculate centroid: {e}")
            return []
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.config.embedding_dimension
