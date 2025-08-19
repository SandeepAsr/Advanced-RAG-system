import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models"""
    openai_model: str = "text-embedding-3-large"
    dimension: int = 3072
    batch_size: int = 100

@dataclass
class VectorStoreConfig:
    """Configuration for vector store"""
    provider: str = "milvus"  # milvus, chromadb, qdrant
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "rag_documents"
    index_type: str = "HNSW"
    metric_type: str = "COSINE"
    ef_construction: int = 200
    m: int = 16

@dataclass
class ChunkingConfig:
    """Configuration for document chunking"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    use_semantic_chunking: bool = True
    hierarchical_levels: list = None

    def __post_init__(self):
        if self.hierarchical_levels is None:
            self.hierarchical_levels = ["title", "section", "paragraph"]

@dataclass
class RetrievalConfig:
    """Configuration for retrieval pipeline"""
    top_k: int = 20
    rerank_top_k: int = 5
    use_multi_query: bool = True
    num_queries: int = 3
    use_hyde: bool = True
    hybrid_alpha: float = 0.5  # Balance between vector and keyword search

@dataclass
class LLMConfig:
    """Configuration for Language Model"""
    model_name: str = "gpt-4-turbo-preview"
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout: int = 60

@dataclass
class CacheConfig:
    """Configuration for caching"""
    use_cache: bool = True
    cache_type: str = "redis"  # redis, memory
    ttl: int = 3600  # Time to live in seconds

@dataclass
class RAGConfig:
    """Main RAG system configuration"""
    embedding: EmbeddingConfig = EmbeddingConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    llm: LLMConfig = LLMConfig()
    cache: CacheConfig = CacheConfig()

    # API Keys
    openai_api_key: Optional[str] = None

    # Paths
    data_path: str = "data"
    pdf_path: str = "data/pdfs"
    chunk_path: str = "data/chunks"

    def __post_init__(self):
        # Load API keys from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # Ensure paths exist
        Path(self.data_path).mkdir(exist_ok=True)
        Path(self.pdf_path).mkdir(exist_ok=True)
        Path(self.chunk_path).mkdir(exist_ok=True)

    def validate(self) -> bool:
        """Validate configuration settings"""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return True

# Global config instance
config = RAGConfig()
