# Create vector database module
vector_db_content = '''"""Vector database implementations for RAG system"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np
import json
from dataclasses import asdict

try:
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from ..ingestion.document_processor import DocumentChunk
from ..utils.helpers import logger, generate_hash
from ..config.config import config

class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    async def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to the vector store"""
        pass
    
    @abstractmethod
    async def search(self, query_embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    async def delete_collection(self) -> None:
        """Delete the entire collection"""
        pass
    
    @abstractmethod
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        pass

class MilvusVectorStore(VectorStore):
    """Milvus vector store implementation"""
    
    def __init__(self):
        if not MILVUS_AVAILABLE:
            raise ImportError("pymilvus is required for MilvusVectorStore")
        
        self.collection_name = config.vector_store.collection_name
        self.dimension = config.embedding.dimension
        self.collection = None
        self._setup_connection()
    
    def _setup_connection(self):
        """Setup connection to Milvus"""
        try:
            connections.connect(
                alias="default",
                host=config.vector_store.host,
                port=config.vector_store.port
            )
            logger.info("Connected to Milvus")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _create_collection(self):
        """Create collection with schema"""
        # Define fields
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="chunk_index", dtype=DataType.INT32),
            FieldSchema(name="tokens", dtype=DataType.INT32)
        ]
        
        # Create schema
        schema = CollectionSchema(fields, description="RAG document chunks")
        
        # Create collection
        self.collection = Collection(self.collection_name, schema)
        
        # Create index
        index_params = {
            "metric_type": config.vector_store.metric_type,
            "index_type": config.vector_store.index_type,
            "params": {
                "M": config.vector_store.m,
                "efConstruction": config.vector_store.ef_construction
            }
        }
        
        self.collection.create_index(field_name="embedding", index_params=index_params)
        logger.info(f"Created Milvus collection: {self.collection_name}")
    
    async def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to Milvus"""
        if not self.collection:
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
            else:
                self._create_collection()
        
        # Prepare data
        ids = [chunk.chunk_id for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks if chunk.embedding is not None]
        contents = [chunk.content for chunk in chunks]
        metadatas = [json.dumps(chunk.metadata) for chunk in chunks]
        sources = [chunk.source for chunk in chunks]
        chunk_indices = [chunk.chunk_index for chunk in chunks]
        tokens = [chunk.tokens for chunk in chunks]
        
        if not embeddings or len(embeddings) != len(chunks):
            raise ValueError("All chunks must have embeddings")
        
        # Insert data
        data = [ids, embeddings, contents, metadatas, sources, chunk_indices, tokens]
        self.collection.insert(data)
        self.collection.flush()
        
        logger.info(f"Added {len(chunks)} documents to Milvus")
    
    async def search(self, query_embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar documents in Milvus"""
        if not self.collection:
            self.collection = Collection(self.collection_name)
        
        self.collection.load()
        
        # Search parameters
        search_params = {"metric_type": config.vector_store.metric_type, "params": {"ef": 64}}
        
        # Perform search
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["content", "metadata", "source", "chunk_index", "tokens"]
        )
        
        # Format results
        formatted_results = []
        for hit in results[0]:
            result = {
                "id": hit.id,
                "score": hit.score,
                "content": hit.entity.get("content"),
                "metadata": json.loads(hit.entity.get("metadata", "{}")),
                "source": hit.entity.get("source"),
                "chunk_index": hit.entity.get("chunk_index"),
                "tokens": hit.entity.get("tokens")
            }
            formatted_results.append(result)
        
        return formatted_results
    
    async def delete_collection(self) -> None:
        """Delete collection"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if not self.collection:
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
            else:
                return {"exists": False}
        
        stats = {
            "exists": True,
            "name": self.collection_name,
            "num_entities": self.collection.num_entities,
            "description": self.collection.description
        }
        
        return stats

class ChromaVectorStore(VectorStore):
    """ChromaDB vector store implementation"""
    
    def __init__(self):
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb is required for ChromaVectorStore")
        
        self.client = chromadb.Client()
        self.collection_name = config.vector_store.collection_name
        self.collection = None
        self._setup_collection()
    
    def _setup_collection(self):
        """Setup ChromaDB collection"""
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Setup ChromaDB collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to setup ChromaDB collection: {e}")
            raise
    
    async def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to ChromaDB"""
        if not chunks:
            return
        
        ids = [chunk.chunk_id for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks if chunk.embedding is not None]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                **chunk.metadata,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "tokens": chunk.tokens
            } for chunk in chunks
        ]
        
        if not embeddings or len(embeddings) != len(chunks):
            raise ValueError("All chunks must have embeddings")
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} documents to ChromaDB")
    
    async def search(self, query_embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar documents in ChromaDB"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        formatted_results = []
        for i in range(len(results['ids'][0])):
            result = {
                "id": results['ids'][0][i],
                "score": 1 - results['distances'][0][i],  # Convert distance to similarity
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "source": results['metadatas'][0][i].get("source"),
                "chunk_index": results['metadatas'][0][i].get("chunk_index"),
                "tokens": results['metadatas'][0][i].get("tokens")
            }
            formatted_results.append(result)
        
        return formatted_results
    
    async def delete_collection(self) -> None:
        """Delete collection"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted ChromaDB collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Could not delete collection: {e}")
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {
                "exists": True,
                "name": self.collection_name,
                "num_entities": count
            }
        except Exception as e:
            logger.error(f"Could not get collection stats: {e}")
            return {"exists": False}

class FAISSVectorStore(VectorStore):
    """FAISS vector store implementation (for local development)"""
    
    def __init__(self):
        if not FAISS_AVAILABLE:
            raise ImportError("faiss is required for FAISSVectorStore")
        
        self.dimension = config.embedding.dimension
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.chunks_metadata = []  # Store metadata separately
        self.id_to_idx = {}  # Map chunk IDs to FAISS indices
        
        logger.info("Initialized FAISS vector store")
    
    async def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to FAISS index"""
        if not chunks:
            return
        
        embeddings = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.chunk_id} has no embedding")
            
            # Normalize embedding for cosine similarity
            embedding = np.array(chunk.embedding, dtype=np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
            
            # Store metadata
            metadata = {
                "id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "tokens": chunk.tokens
            }
            
            self.id_to_idx[chunk.chunk_id] = len(self.chunks_metadata)
            self.chunks_metadata.append(metadata)
        
        # Add to FAISS index
        embeddings_array = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings_array)
        
        logger.info(f"Added {len(chunks)} documents to FAISS index")
    
    async def search(self, query_embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar documents in FAISS"""
        # Normalize query embedding
        query_embedding = np.array(query_embedding, dtype=np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks_metadata):
                result = {
                    **self.chunks_metadata[idx],
                    "score": float(score)
                }
                results.append(result)
        
        return results
    
    async def delete_collection(self) -> None:
        """Reset the FAISS index"""
        self.index.reset()
        self.chunks_metadata.clear()
        self.id_to_idx.clear()
        logger.info("Reset FAISS index")
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get FAISS index statistics"""
        return {
            "exists": True,
            "name": "faiss_index",
            "num_entities": self.index.ntotal,
            "dimension": self.dimension
        }

def get_vector_store() -> VectorStore:
    """Factory function to get vector store based on configuration"""
    provider = config.vector_store.provider.lower()
    
    if provider == "milvus":
        return MilvusVectorStore()
    elif provider == "chromadb":
        return ChromaVectorStore()
    elif provider == "faiss":
        return FAISSVectorStore()
    else:
        raise ValueError(f"Unsupported vector store provider: {provider}")
'''

with open(base_dir / "src" / "database" / "vector_store.py", "w") as f:
    f.write(vector_db_content)
    
print("Vector database module created!")