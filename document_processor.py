"""Document ingestion and processing module"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import re
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

import PyPDF2
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from openai import OpenAI

from ..utils.helpers import clean_text, count_tokens, generate_hash, logger
from ..config.config import config

@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    source: str
    chunk_index: int
    tokens: int
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class DocumentProcessor(ABC):
    """Abstract base class for document processors"""

    @abstractmethod
    def extract_text(self, file_path: Path) -> str:
        pass

    @abstractmethod
    def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        pass

class PDFProcessor(DocumentProcessor):
    """PDF document processor"""

    def extract_text(self, file_path: Path) -> str:
        """Extract text from PDF using multiple methods"""
        text = ""

        # Try pdfplumber first (better for complex layouts)
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber failed for {file_path}: {e}")

            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e:
                logger.error(f"PyPDF2 also failed for {file_path}: {e}")
                raise

        return clean_text(text)

    def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from PDF"""
        metadata = {
            "filename": file_path.name,
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "created_at": file_path.stat().st_ctime,
            "modified_at": file_path.stat().st_mtime
        }

        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                metadata.update({
                    "num_pages": len(reader.pages),
                    "title": reader.metadata.get('/Title', '') if reader.metadata else '',
                    "author": reader.metadata.get('/Author', '') if reader.metadata else '',
                    "subject": reader.metadata.get('/Subject', '') if reader.metadata else ''
                })
        except Exception as e:
            logger.warning(f"Could not extract PDF metadata: {e}")

        return metadata

class SemanticChunker:
    """Semantic-aware document chunker"""

    def __init__(self, client: OpenAI):
        self.client = client
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

    async def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk document using hierarchical and semantic approaches"""
        chunks = []

        if config.chunking.use_semantic_chunking:
            chunks = await self._semantic_chunking(text, metadata)
        else:
            chunks = self._hierarchical_chunking(text, metadata)

        return chunks

    def _hierarchical_chunking(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Hierarchical chunking based on document structure"""
        # Split by hierarchical levels
        sections = self._identify_sections(text)
        chunks = []

        for section_idx, section in enumerate(sections):
            # Further split each section
            section_chunks = self.text_splitter.split_text(section['content'])

            for chunk_idx, chunk_text in enumerate(section_chunks):
                chunk = DocumentChunk(
                    content=chunk_text,
                    metadata={
                        **metadata,
                        "section": section['type'],
                        "section_index": section_idx,
                        "hierarchy_level": section.get('level', 0)
                    },
                    chunk_id=generate_hash(chunk_text),
                    source=metadata['filename'],
                    chunk_index=len(chunks),
                    tokens=count_tokens(chunk_text)
                )
                chunks.append(chunk)

        return chunks

    async def _semantic_chunking(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Semantic chunking using AI to identify coherent segments"""
        # First do hierarchical chunking
        initial_chunks = self._hierarchical_chunking(text, metadata)

        # Then refine using semantic analysis
        refined_chunks = []

        for chunk in initial_chunks:
            # Use LLM to identify if chunk contains complete ideas
            if len(chunk.content) > config.chunking.chunk_size * 1.5:
                sub_chunks = await self._split_semantically(chunk.content, metadata)
                refined_chunks.extend(sub_chunks)
            else:
                refined_chunks.append(chunk)

        # Update chunk indices
        for i, chunk in enumerate(refined_chunks):
            chunk.chunk_index = i

        return refined_chunks

    async def _split_semantically(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Split text based on semantic boundaries"""
        prompt = f"""
        Split the following text into semantically coherent chunks. Each chunk should:
        1. Contain a complete idea or concept
        2. Be roughly {config.chunking.chunk_size} characters
        3. Have natural break points

        Text: {text[:2000]}...

        Return the split points as line numbers.
        """

        try:
            response = await self.client.chat.completions.acreate(
                model=config.llm.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )

            # Parse response and split accordingly
            # For now, fall back to rule-based splitting
            return self._rule_based_semantic_split(text, metadata)

        except Exception as e:
            logger.warning(f"Semantic splitting failed: {e}")
            return self._rule_based_semantic_split(text, metadata)

    def _rule_based_semantic_split(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Rule-based semantic splitting"""
        # Split on paragraph boundaries first
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) < config.chunking.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunk = DocumentChunk(
                        content=current_chunk.strip(),
                        metadata=metadata,
                        chunk_id=generate_hash(current_chunk),
                        source=metadata['filename'],
                        chunk_index=len(chunks),
                        tokens=count_tokens(current_chunk)
                    )
                    chunks.append(chunk)
                current_chunk = para + "\n\n"

        # Add final chunk
        if current_chunk:
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                metadata=metadata,
                chunk_id=generate_hash(current_chunk),
                source=metadata['filename'],
                chunk_index=len(chunks),
                tokens=count_tokens(current_chunk)
            )
            chunks.append(chunk)

        return chunks

    def _identify_sections(self, text: str) -> List[Dict[str, Any]]:
        """Identify document sections based on patterns"""
        sections = []

        # Patterns for different section types
        patterns = {
            'title': r'^[A-Z][^\n]*$',
            'heading': r'^#+\s+.*$',
            'paragraph': r'^[^\n]+(?:\n[^\n]+)*$'
        }

        lines = text.split('\n')
        current_section = {'type': 'paragraph', 'content': '', 'level': 0}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line matches any pattern
            section_type = 'paragraph'
            level = 0

            if re.match(patterns['title'], line) and len(line) < 100:
                section_type = 'title'
                level = 1
            elif re.match(patterns['heading'], line):
                section_type = 'heading'
                level = len(line) - len(line.lstrip('#'))

            # If section type changed, save current and start new
            if section_type != current_section['type'] and current_section['content']:
                sections.append(current_section)
                current_section = {'type': section_type, 'content': line + '\n', 'level': level}
            else:
                current_section['content'] += line + '\n'

        # Add final section
        if current_section['content']:
            sections.append(current_section)

        return sections

class DocumentIngestion:
    """Main document ingestion pipeline"""

    def __init__(self):
        self.client = OpenAI(api_key=config.openai_api_key)
        self.processors = {
            '.pdf': PDFProcessor()
        }
        self.chunker = SemanticChunker(self.client)

    async def process_documents(self, file_paths: List[Path]) -> List[DocumentChunk]:
        """Process multiple documents"""
        all_chunks = []

        for file_path in file_paths:
            try:
                chunks = await self.process_single_document(file_path)
                all_chunks.extend(chunks)
                logger.info(f"Processed {file_path}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")

        return all_chunks

    async def process_single_document(self, file_path: Path) -> List[DocumentChunk]:
        """Process a single document"""
        extension = file_path.suffix.lower()

        if extension not in self.processors:
            raise ValueError(f"Unsupported file type: {extension}")

        processor = self.processors[extension]

        # Extract text and metadata
        text = processor.extract_text(file_path)
        metadata = processor.extract_metadata(file_path)

        # Chunk the document
        chunks = await self.chunker.chunk_document(text, metadata)

        return chunks

    async def save_chunks(self, chunks: List[DocumentChunk], output_path: Path) -> None:
        """Save chunks to JSON file"""
        chunks_data = [chunk.to_dict() for chunk in chunks]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(chunks)} chunks to {output_path}")
