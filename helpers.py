"""Utility functions for RAG system"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import tiktoken
import re
from datetime import datetime

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text for given model"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback estimation
        return len(text.split()) * 1.3

def generate_hash(text: str) -> str:
    """Generate MD5 hash for text"""
    return hashlib.md5(text.encode()).hexdigest()

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:-]', '', text)
    return text.strip()

def chunk_text_by_sentences(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Chunk text by sentence boundaries"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Add overlap
    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0 and overlap > 0:
            prev_words = chunks[i-1].split()[-overlap:]
            chunk = " ".join(prev_words) + " " + chunk
        overlapped_chunks.append(chunk)

    return overlapped_chunks

def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Save data to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(filepath: Path) -> Dict[str, Any]:
    """Load data from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

class TimestampedCache:
    """Simple in-memory cache with timestamps"""

    def __init__(self, ttl: int = 3600):
        self.cache = {}
        self.ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now().timestamp() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set item in cache"""
        self.cache[key] = (value, datetime.now().timestamp())

    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()

def extract_metadata_from_filename(filename: str) -> Dict[str, str]:
    """Extract metadata from filename"""
    metadata = {
        "filename": filename,
        "extension": Path(filename).suffix,
        "name": Path(filename).stem
    }
    return metadata

logger = setup_logging()
