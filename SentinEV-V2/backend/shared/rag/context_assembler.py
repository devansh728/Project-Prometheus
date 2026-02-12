"""
Context Assembler for Agent Prompts
Builds rich context from retrieved documents for LLM prompts.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from shared.rag.retrieval_service import RetrievalResult, retrieval_service


@dataclass
class AssembledContext:
    """Assembled context for an agent prompt."""

    text: str
    sources: List[str]
    categories: List[str]
    token_estimate: int


class ContextAssembler:
    """
    Assembles retrieved documents into structured context for LLM prompts.
    """

    # Approximate tokens per character (conservative estimate)
    CHARS_PER_TOKEN = 4

    # Context templates
    TEMPLATES = {
        "diagnosis": """
## Relevant Repair Information

{repair_context}

## Related CAPA Patterns

{capa_context}
""",
        "general": """
## Knowledge Base Context

{context}
""",
        "scheduling": """
## Service Information

{context}
""",
    }

    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
        self.max_chars = max_tokens * self.CHARS_PER_TOKEN

    def assemble_for_diagnosis(
        self,
        failure_type: str,
        symptoms: List[str],
        vehicle_info: Optional[Dict] = None,
    ) -> AssembledContext:
        """
        Assemble context for a diagnosis query.
        """
        # Retrieve repair information
        repair_results = retrieval_service.retrieve_for_failure_type(
            failure_type=failure_type, vehicle_info=vehicle_info
        )

        # Retrieve CAPA patterns
        capa_results = retrieval_service.retrieve_capa_patterns(
            symptoms=symptoms, vehicle_info=vehicle_info
        )

        # Build repair context
        repair_parts = []
        for r in repair_results[:3]:
            repair_parts.append(f"### {r.source} - {r.section}\n{r.content}")
        repair_context = (
            "\n\n".join(repair_parts)
            if repair_parts
            else "No specific repair information found."
        )

        # Build CAPA context
        capa_parts = []
        for r in capa_results[:2]:
            capa_parts.append(f"### {r.source}\n{r.content}")
        capa_context = (
            "\n\n".join(capa_parts)
            if capa_parts
            else "No matching CAPA patterns found."
        )

        # Assemble full context
        full_text = self.TEMPLATES["diagnosis"].format(
            repair_context=repair_context, capa_context=capa_context
        )

        # Truncate if needed
        if len(full_text) > self.max_chars:
            full_text = full_text[: self.max_chars] + "\n... [truncated]"

        # Collect sources
        all_results = repair_results + capa_results
        sources = list(set(r.source for r in all_results))
        categories = list(set(r.category for r in all_results))

        return AssembledContext(
            text=full_text,
            sources=sources,
            categories=categories,
            token_estimate=len(full_text) // self.CHARS_PER_TOKEN,
        )

    def assemble_for_query(
        self, query: str, context_type: str = "general"
    ) -> AssembledContext:
        """
        Assemble context for a general query.
        """
        results = retrieval_service.retrieve(
            query=query, query_type=context_type, n_results=5
        )

        # Build context
        parts = []
        for r in results:
            header = f"### {r.source}"
            if r.section:
                header += f" - {r.section}"
            parts.append(f"{header}\n{r.content}")

        context = (
            "\n\n".join(parts)
            if parts
            else "No relevant information found in knowledge base."
        )

        template = self.TEMPLATES.get(context_type, self.TEMPLATES["general"])
        full_text = template.format(context=context)

        # Truncate if needed
        if len(full_text) > self.max_chars:
            full_text = full_text[: self.max_chars] + "\n... [truncated]"

        sources = list(set(r.source for r in results))
        categories = list(set(r.category for r in results))

        return AssembledContext(
            text=full_text,
            sources=sources,
            categories=categories,
            token_estimate=len(full_text) // self.CHARS_PER_TOKEN,
        )


# Singleton
context_assembler = ContextAssembler()
