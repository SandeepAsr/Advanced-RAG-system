# Create embedding service
embedding_content = '''"""Embedding service for document and query embeddings"""

import asyncio
from typing import List, Dict, Any, Optional, Union
import numpy as np
from openai import AsyncOpenAI
import tiktoken

from ..utils.helpers import logger, count_tokens
from ..config.config import config

class EmbeddingService:
    """Service for generating embeddings using OpenAI API"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.model = config.embedding.openai_model
        self.dimension = config.embedding.dimension
        self.batch_size = config.embedding.batch_size
        
        # Token encoder for the embedding model
        try:
            self.encoder = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")
    
    async def embed_texts(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if not texts:
            return []
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            if show_progress:
                logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}")
            
            try:
                batch_embeddings = await self._embed_batch(batch)
                all_embeddings.extend(batch_embeddings)
                
                # Small delay to respect rate limits
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to embed batch {i//self.batch_size + 1}: {e}")
                # Add empty embeddings for failed batch
                all_embeddings.extend([[] for _ in batch])
        
        return all_embeddings
    
    async def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        embeddings = await self.embed_texts([text], show_progress=False)
        return embeddings[0] if embeddings else []
    
    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        # Check token limits
        processed_texts = []
        for text in texts:
            token_count = len(self.encoder.encode(text))
            if token_count > 8000:  # Conservative limit for embedding models
                # Truncate text
                tokens = self.encoder.encode(text)[:8000]
                text = self.encoder.decode(tokens)
                logger.warning(f"Truncated text with {token_count} tokens to 8000 tokens")
            processed_texts.append(text)
        
        try:
            response = await self.client.embeddings.create(
                input=processed_texts,
                model=self.model
            )
            
            embeddings = []
            for item in response.data:
                embeddings.append(item.embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"OpenAI embedding API error: {e}")
            raise
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """Validate embedding format and dimension"""
        if not embedding:
            return False
        
        if len(embedding) != self.dimension:
            logger.warning(f"Embedding dimension mismatch: expected {self.dimension}, got {len(embedding)}")
            return False
        
        # Check if embedding is valid (not all zeros)
        if all(x == 0 for x in embedding):
            logger.warning("Embedding is all zeros")
            return False
        
        return True
    
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        if not self.validate_embedding(embedding1) or not self.validate_embedding(embedding2):
            return 0.0
        
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def get_embedding_stats(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        """Get statistics about embeddings"""
        if not embeddings:
            return {"count": 0}
        
        valid_embeddings = [emb for emb in embeddings if self.validate_embedding(emb)]
        
        if not valid_embeddings:
            return {"count": 0, "valid_count": 0}
        
        # Convert to numpy array for calculations
        emb_array = np.array(valid_embeddings)
        
        stats = {
            "count": len(embeddings),
            "valid_count": len(valid_embeddings),
            "dimension": emb_array.shape[1] if emb_array.size > 0 else 0,
            "mean_magnitude": float(np.mean(np.linalg.norm(emb_array, axis=1))),
            "std_magnitude": float(np.std(np.linalg.norm(emb_array, axis=1))),
        }
        
        return stats

# Global embedding service instance
embedding_service = EmbeddingService()
'''

with open(base_dir / "src" / "database" / "embeddings.py", "w") as f:
    f.write(embedding_content)
    
print("Embedding service created!")