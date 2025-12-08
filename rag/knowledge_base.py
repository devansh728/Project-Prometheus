"""
SentinelEY - RAG Knowledge Base
Embeds service manuals and CAPA records for agent retrieval
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print(
        "⚠️  LangChain not installed. Run: pip install langchain langchain-community chromadb"
    )


class DiagnosticsRAG:
    """
    RAG system for service manuals, CAPA records, and DTC codes.
    Uses ChromaDB for vector storage and HuggingFace embeddings.
    """

    PERSIST_DIR = "data/vectordb"

    def __init__(self, persist_directory: str = None):
        """Initialize RAG system."""
        self.persist_directory = persist_directory or self.PERSIST_DIR
        self.embeddings = None
        self.vectorstore = None
        self.collections = {}

        if LANGCHAIN_AVAILABLE:
            self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embedding model."""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )
            print("✅ Embeddings initialized (all-MiniLM-L6-v2)")
        except Exception as e:
            print(f"⚠️  Embeddings init failed: {e}")
            self.embeddings = None

    def build_knowledge_base(self, data_dir: str = "data/knowledge"):
        """
        Build the vector database from knowledge files.
        Should be run once to index all documents.
        """
        if not LANGCHAIN_AVAILABLE or self.embeddings is None:
            print("❌ Cannot build knowledge base - LangChain/embeddings not available")
            return

        os.makedirs(self.persist_directory, exist_ok=True)

        all_documents = []

        # Load service manuals
        manuals_path = Path(data_dir) / "service_manuals.json"
        if manuals_path.exists():
            manuals = self._load_service_manuals(manuals_path)
            all_documents.extend(manuals)
            print(f"📚 Loaded {len(manuals)} service manual documents")

        # Load CAPA records
        capa_path = Path(data_dir) / "capa_records.json"
        if capa_path.exists():
            capa_docs = self._load_capa_records(capa_path)
            all_documents.extend(capa_docs)
            print(f"📋 Loaded {len(capa_docs)} CAPA record documents")

        if not all_documents:
            print("⚠️  No documents found to index")
            return

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ". ", " "]
        )
        split_docs = text_splitter.split_documents(all_documents)
        print(f"📄 Split into {len(split_docs)} chunks")

        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name="diagnostics_kb",
        )

        print(f"✅ Knowledge base built and persisted to {self.persist_directory}")

    def _load_service_manuals(self, path: Path) -> List[Document]:
        """Load and convert service manuals to documents."""
        documents = []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for manual in data.get("service_manuals", []):
            doc = Document(
                page_content=manual["content"],
                metadata={
                    "source": "service_manual",
                    "id": manual["id"],
                    "title": manual["title"],
                    "category": manual["category"],
                },
            )
            documents.append(doc)

        return documents

    def _load_capa_records(self, path: Path) -> List[Document]:
        """Load and convert CAPA records to documents."""
        documents = []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # CAPA records
        for record in data.get("capa_records", []):
            content = f"""CAPA ID: {record['capa_id']}
Failure Mode: {record['failure_mode']}
Status: {record['status']}
Affected Models: {', '.join(record['affected_models'])}
Affected Batches: {', '.join(record['affected_batches'])}
Occurrence Count: {record['occurrence_count']}

Root Cause Analysis:
{record['root_cause_analysis']['description']}
Category: {record['root_cause_analysis']['category']}

5-Why Analysis:
{chr(10).join(record['root_cause_analysis']['five_why'])}

Corrective Actions:
{chr(10).join('- ' + action for action in record['corrective_actions'])}

Preventive Actions:
{chr(10).join('- ' + action for action in record['preventive_actions'])}

Cost Impact: ₹{record['cost_impact_inr']:,}
Customer Impact: {record['customer_impact']}"""

            doc = Document(
                page_content=content,
                metadata={
                    "source": "capa_record",
                    "capa_id": record["capa_id"],
                    "status": record["status"],
                    "failure_mode": record["failure_mode"],
                },
            )
            documents.append(doc)

        # DTC codes
        for dtc in data.get("dtc_codes", []):
            content = f"""DTC Code: {dtc['code']}
Description: {dtc['description']}
Severity: {dtc['severity']}
Category: {dtc['category']}"""

            doc = Document(
                page_content=content,
                metadata={
                    "source": "dtc_code",
                    "code": dtc["code"],
                    "severity": dtc["severity"],
                    "category": dtc["category"],
                },
            )
            documents.append(doc)

        return documents

    def load_knowledge_base(self):
        """Load existing vector database."""
        if not LANGCHAIN_AVAILABLE or self.embeddings is None:
            print("❌ Cannot load knowledge base - LangChain/embeddings not available")
            return False

        if not os.path.exists(self.persist_directory):
            print("⚠️  Knowledge base not found. Run build_knowledge_base() first.")
            return False

        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="diagnostics_kb",
        )
        print(f"✅ Knowledge base loaded from {self.persist_directory}")
        return True

    def query_manuals(
        self, symptom: str, vehicle_model: str = None, k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search service manuals for diagnostic procedures.

        Args:
            symptom: Symptom or issue description
            vehicle_model: Optional vehicle model filter
            k: Number of results to return
        """
        if self.vectorstore is None:
            return []

        query = symptom
        if vehicle_model:
            query = f"{vehicle_model} {symptom}"

        # Search with filter
        filter_dict = {"source": "service_manual"}

        docs = self.vectorstore.similarity_search(query, k=k, filter=filter_dict)

        return [
            {
                "content": doc.page_content,
                "title": doc.metadata.get("title"),
                "category": doc.metadata.get("category"),
                "relevance_score": 1.0,  # ChromaDB doesn't return scores by default
            }
            for doc in docs
        ]

    def find_similar_failures(
        self, anomaly_type: str, k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find historical CAPA records with similar failure patterns.

        Args:
            anomaly_type: Type of anomaly detected
            k: Number of results
        """
        if self.vectorstore is None:
            return []

        filter_dict = {"source": "capa_record"}

        docs = self.vectorstore.similarity_search(anomaly_type, k=k, filter=filter_dict)

        return [
            {
                "content": doc.page_content,
                "capa_id": doc.metadata.get("capa_id"),
                "failure_mode": doc.metadata.get("failure_mode"),
                "status": doc.metadata.get("status"),
            }
            for doc in docs
        ]

    def get_dtc_info(self, dtc_code: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific DTC code.

        Args:
            dtc_code: The diagnostic trouble code (e.g., "P0A1F")
        """
        if self.vectorstore is None:
            return None

        docs = self.vectorstore.similarity_search(
            f"DTC Code: {dtc_code}", k=1, filter={"source": "dtc_code"}
        )

        if docs:
            return {
                "content": docs[0].page_content,
                "code": docs[0].metadata.get("code"),
                "severity": docs[0].metadata.get("severity"),
                "category": docs[0].metadata.get("category"),
            }
        return None

    def get_rca_insights(self, failure_mode: str) -> Dict[str, Any]:
        """
        Get Root Cause Analysis insights for a failure mode.

        Args:
            failure_mode: Description of the failure
        """
        similar = self.find_similar_failures(failure_mode, k=3)

        if not similar:
            return {"found": False, "message": "No similar historical failures found"}

        # Extract insights from similar records
        return {
            "found": True,
            "similar_records": similar,
            "recommendation": f"Review {len(similar)} similar historical failures for guidance",
        }

    def semantic_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        General semantic search across all knowledge.

        Args:
            query: Natural language query
            k: Number of results
        """
        if self.vectorstore is None:
            return []

        docs = self.vectorstore.similarity_search(query, k=k)

        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source"),
                "metadata": doc.metadata,
            }
            for doc in docs
        ]


def build_knowledge_base():
    """CLI function to build the knowledge base."""
    print("🔨 Building SentinelEY Knowledge Base")
    print("=" * 50)

    rag = DiagnosticsRAG()
    rag.build_knowledge_base()

    print("\n✅ Knowledge base ready!")
    print("You can now use RAG queries in the agents.")


if __name__ == "__main__":
    import sys

    if "--build" in sys.argv:
        build_knowledge_base()
    else:
        # Test queries
        print("🔍 Testing RAG System")
        print("=" * 50)

        rag = DiagnosticsRAG()

        if rag.load_knowledge_base():
            # Test manual search
            results = rag.query_manuals("battery cell imbalance")
            print(f"\n📚 Manual search results: {len(results)}")
            for r in results:
                print(f"  - {r['title']}: {r['content'][:100]}...")

            # Test CAPA search
            results = rag.find_similar_failures("thermal runaway")
            print(f"\n📋 CAPA search results: {len(results)}")
            for r in results:
                print(f"  - {r['capa_id']}: {r['failure_mode']}")
        else:
            print("Run with --build flag first to create knowledge base")
