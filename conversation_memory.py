"""Conversational Memory Management"""

from typing import List, Dict, Any
import numpy as np
from ..database.embeddings import embedding_service
from ..utils.helpers import logger, generate_hash

class ConversationMemoryStore:
    """Store and retrieve past conversational snippets using vector search"""
    def __init__(self):
        self.memory = []
        self.embeddings = []
        self.memory_hash_map = {}

    async def add_snippet(self, text: str, metadata: Dict[str, Any]):
        emb = await embedding_service.embed_single(text)
        hash_id = generate_hash(text)
        self.memory.append({"text": text, "metadata": metadata, "id": hash_id})
        self.embeddings.append(emb)
        self.memory_hash_map[hash_id] = len(self.memory) - 1

    async def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        q_emb = await embedding_service.embed_single(query)
        if not self.embeddings:
            return []
        scores = np.dot(np.array(self.embeddings), np.array(q_emb))
        idx_sorted = np.argsort(-scores)[:top_k]
        results = [self.memory[i] for i in idx_sorted]
        return results

    def clear(self):
        self.memory.clear()
        self.embeddings.clear()
        self.memory_hash_map.clear()

# Standalone LLM-based query transformer
import asyncio
from openai import AsyncOpenAI
from ..config.config import config
class QueryTransformer:
    """Use LLM to transform follow-up queries into standalone queries"""
    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.model = config.llm.model_name

    async def transform(self, followup: str, history: List[Dict[str, Any]]) -> str:
        context = "
".join([f"Q: {h['question']}  A: {h['answer']}" for h in history[-5:]])
        prompt = f"""
        Here is a conversation history:
{context}
Now rewrite the user follow-up question as a fully standalone question, in context, with all necessary details.
Follow-up: {followup}
Standalone question:"
        try:
            result = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=256
            )
            return result.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Query transformation failed: {e}")
            return followup
