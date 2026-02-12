"""
Document Chunking & Embedding Pipeline
Processes knowledge documents for RAG retrieval.
"""

from typing import List, Dict, Any, Generator
from pathlib import Path
import re


class DocumentChunker:
    """
    Chunks documents into smaller pieces for embedding.
    Supports markdown documents with section-aware chunking.
    """

    # Target chunk sizes
    DEFAULT_CHUNK_SIZE = 500  # characters
    DEFAULT_OVERLAP = 50

    def __init__(
        self, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_markdown(self, content: str, source: str) -> List[Dict[str, Any]]:
        """
        Chunk a markdown document, respecting section boundaries.

        Returns:
            List of chunks with metadata
        """
        chunks = []

        # Split by headers
        sections = self._split_by_headers(content)

        for section in sections:
            section_title = section.get("title", "")
            section_content = section.get("content", "")
            section_level = section.get("level", 1)

            # If section is small enough, keep as one chunk
            if len(section_content) <= self.chunk_size:
                if section_content.strip():
                    chunks.append(
                        {
                            "content": section_content.strip(),
                            "metadata": {
                                "source": source,
                                "section": section_title,
                                "level": section_level,
                                "category": self._infer_category(source),
                            },
                        }
                    )
            else:
                # Split large sections into overlapping chunks
                sub_chunks = self._split_with_overlap(section_content)
                for i, sub_chunk in enumerate(sub_chunks):
                    chunks.append(
                        {
                            "content": sub_chunk.strip(),
                            "metadata": {
                                "source": source,
                                "section": section_title,
                                "level": section_level,
                                "chunk_index": i,
                                "category": self._infer_category(source),
                            },
                        }
                    )

        return chunks

    def _split_by_headers(self, content: str) -> List[Dict[str, Any]]:
        """Split markdown content by headers."""
        sections = []

        # Pattern to match markdown headers
        header_pattern = r"^(#{1,6})\s+(.+)$"

        lines = content.split("\n")
        current_section = {"title": "Introduction", "content": "", "level": 0}

        for line in lines:
            match = re.match(header_pattern, line)
            if match:
                # Save previous section if has content
                if current_section["content"].strip():
                    sections.append(current_section)

                # Start new section
                level = len(match.group(1))
                title = match.group(2).strip()
                current_section = {"title": title, "content": "", "level": level}
            else:
                current_section["content"] += line + "\n"

        # Add last section
        if current_section["content"].strip():
            sections.append(current_section)

        return sections

    def _split_with_overlap(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end within last 100 chars
                search_start = max(end - 100, start)
                sentence_ends = [
                    text.rfind(". ", search_start, end),
                    text.rfind("? ", search_start, end),
                    text.rfind("! ", search_start, end),
                    text.rfind("\n", search_start, end),
                ]
                best_end = (
                    max(e for e in sentence_ends if e > 0)
                    if any(e > 0 for e in sentence_ends)
                    else end
                )
                if best_end > start:
                    end = best_end + 1

            chunks.append(text[start:end])
            start = end - self.overlap

        return chunks

    def _infer_category(self, source: str) -> str:
        """Infer document category from source path."""
        source_lower = source.lower()

        if "repair" in source_lower or "guide" in source_lower:
            return "repair_guide"
        elif "capa" in source_lower:
            return "capa"
        elif "manual" in source_lower:
            return "manual"
        elif "diagnostic" in source_lower or "dtc" in source_lower:
            return "diagnostic"
        else:
            return "general"


class EmbeddingPipeline:
    """
    Orchestrates document processing and embedding.
    """

    def __init__(self, vector_store, chunker: DocumentChunker = None):
        self.vector_store = vector_store
        self.chunker = chunker or DocumentChunker()

    def process_directory(
        self, directory: str, extensions: List[str] = [".md", ".txt"]
    ) -> Dict[str, int]:
        """
        Process all documents in a directory.

        Returns:
            Stats about processed documents
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            return {"error": f"Directory {directory} not found"}

        total_docs = 0
        total_chunks = 0

        for ext in extensions:
            for file_path in dir_path.rglob(f"*{ext}"):
                chunks = self.process_file(str(file_path))
                total_docs += 1
                total_chunks += chunks

        return {"documents_processed": total_docs, "chunks_created": total_chunks}

    def process_file(self, file_path: str) -> int:
        """
        Process a single file and add to vector store.

        Returns:
            Number of chunks added
        """
        path = Path(file_path)
        if not path.exists():
            return 0

        content = path.read_text(encoding="utf-8")
        relative_path = str(path.name)

        # Chunk the document
        if path.suffix.lower() == ".md":
            chunks = self.chunker.chunk_markdown(content, relative_path)
        else:
            # Plain text: simple chunking
            chunks = [
                {
                    "content": content,
                    "metadata": {
                        "source": relative_path,
                        "category": self.chunker._infer_category(relative_path),
                    },
                }
            ]

        # Add to vector store
        if chunks:
            documents = [c["content"] for c in chunks]
            metadatas = [c["metadata"] for c in chunks]
            return self.vector_store.add_documents(documents, metadatas)

        return 0


# Factory function for pipeline
def create_embedding_pipeline():
    """Create an embedding pipeline with default vector store."""
    from shared.rag.vector_store import vector_store

    return EmbeddingPipeline(vector_store)
