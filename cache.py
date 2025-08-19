"""Cache Layer for RAG System"""

from typing import Any
import redis
import pickle
from ..config.config import config
from ..utils.helpers import logger

class CacheLayer:
    """Cache for frequently asked queries"""
    def __init__(self):
        self.use_cache = config.cache.use_cache
        if config.cache.cache_type == "redis":
            try:
                self.client = redis.Redis(host='localhost', port=6379, db=0)
                self.ready = True
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}")
                self.client = None
                self.ready = False
        else:
            # In-memory fallback
            self.cache = {}
            self.ready = True
    def get(self, key: str) -> Any:
        if not self.use_cache or not self.ready:
            return None
        if hasattr(self, 'client') and self.client:
            val = self.client.get(key)
            return pickle.loads(val) if val else None
        else:
            return self.cache.get(key)
    def set(self, key: str, value: Any):
        if not self.use_cache or not self.ready:
            return
        if hasattr(self, 'client') and self.client:
            self.client.setex(key, config.cache.ttl, pickle.dumps(value))
        else:
            self.cache[key] = value
    def clear(self):
        if hasattr(self, 'client') and self.client:
            self.client.flushdb()
        else:
            self.cache.clear()
