"""
SentinelEY - RAG Knowledge Base for ML Model Training
Uses ChromaDB for vector storage and retrieval of industry faults
"""

from __future__ import annotations

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path

# Type checking imports (not executed at runtime)
if TYPE_CHECKING:
    from langchain_core.documents import Document

# Runtime imports - check for langchain_community specifically
LANGCHAIN_AVAILABLE = False
DocumentClass = None
Chroma = None
HuggingFaceEmbeddings = None

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.documents import Document as DocumentClass

    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ LangChain import error: {e}")
    print("Run: pip install langchain-core langchain-community chromadb")
except Exception as e:
    print(f"⚠️ LangChain import error: {e}")


class MLKnowledgeBase:
    """
    RAG Knowledge Base for ML Model Training.

    Uses ChromaDB to store and retrieve:
    - Industry fault patterns
    - Vehicle manual specifications
    - Historical failure data

    This enables RAG-augmented training for the Digital Twin model.
    """

    COLLECTION_NAME = "ml_knowledge_base"
    PERSIST_DIR = "data/ml_vectordb"

    def __init__(self, persist_directory: str = None):
        """
        Initialize the knowledge base.

        Args:
            persist_directory: Directory to persist ChromaDB
        """
        self.persist_dir = persist_directory or self.PERSIST_DIR
        self.embeddings = None
        self.vectorstore = None
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize the embedding model."""
        if not LANGCHAIN_AVAILABLE:
            return

        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as e:
            print(f"⚠️ Could not initialize embeddings: {e}")

    def build_knowledge_base(
        self,
        faults_csv_path: str = None,
        manual_json_path: str = None,
        faults_json_path: str = None,
    ):
        """
        Build the vector database from knowledge files.

        Args:
            faults_csv_path: Path to industry_faults.csv
            manual_json_path: Path to vehicle_manual.json
            faults_json_path: Path to industry_faults.json
        """
        if not LANGCHAIN_AVAILABLE or not self.embeddings:
            print("❌ Cannot build knowledge base without LangChain")
            return

        documents = []

        # Load industry faults from JSON (preferred)
        if faults_json_path and os.path.exists(faults_json_path):
            fault_docs = self._load_faults_json(faults_json_path)
            documents.extend(fault_docs)
            print(f"✅ Loaded {len(fault_docs)} fault documents from JSON")
        # Fall back to CSV
        elif faults_csv_path and os.path.exists(faults_csv_path):
            fault_docs = self._load_faults_csv(faults_csv_path)
            documents.extend(fault_docs)
            print(f"✅ Loaded {len(fault_docs)} fault documents from CSV")

        # Load vehicle manual
        if manual_json_path and os.path.exists(manual_json_path):
            manual_docs = self._load_manual_json(manual_json_path)
            documents.extend(manual_docs)
            print(f"✅ Loaded {len(manual_docs)} manual documents")

        if not documents:
            print("⚠️ No documents to index")
            return

        # Create vector store
        os.makedirs(self.persist_dir, exist_ok=True)

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.COLLECTION_NAME,
            persist_directory=self.persist_dir,
        )

        print(f"✅ Knowledge base built with {len(documents)} documents")
        print(f"   Persisted to: {self.persist_dir}")

    def _load_faults_json(self, path: str) -> List:
        """
        Load industry faults from JSON file.

        Args:
            path: Path to industry_faults.json

        Returns:
            List of Document objects
        """
        if not DocumentClass:
            return []

        documents = []

        with open(path, "r") as f:
            faults = json.load(f)

        for fault in faults:
            # Create rich text content for embedding
            symptoms = fault.get("symptoms", [])
            if isinstance(symptoms, list):
                symptoms_text = ", ".join(symptoms)
            else:
                symptoms_text = str(symptoms)

            content = f"""
Fault ID: {fault.get('fault_id', 'Unknown')}
Error Code: {fault.get('error_code', 'Unknown')}
Fault Name: {fault.get('fault_name', 'Unknown')}
Component: {fault.get('component', 'Unknown')}
Description: {fault.get('description', '')}
Severity: {fault.get('severity', 'Unknown')}
Symptoms: {symptoms_text}
Recommended Action: {fault.get('recommended_action', '')}
RCA Category: {fault.get('rca_category', '')}
            """.strip()

            doc = DocumentClass(
                page_content=content,
                metadata={
                    "source": "industry_faults",
                    "fault_id": str(fault.get("fault_id", "")),
                    "error_code": str(fault.get("error_code", "")),
                    "component": str(fault.get("component", "")),
                    "severity": str(fault.get("severity", "")),
                    "type": "fault",
                },
            )
            documents.append(doc)

        return documents

    def _load_faults_csv(self, path: str) -> List:
        """
        Load industry faults from CSV.

        Args:
            path: Path to industry_faults.csv

        Returns:
            List of Document objects
        """
        documents = []

        df = pd.read_csv(path)

        for _, row in df.iterrows():
            # Create rich text content for embedding
            content = f"""
Fault ID: {row.get('fault_id', 'Unknown')}
Error Code: {row.get('error_code', 'Unknown')}
Component: {row.get('component', 'Unknown')}
Description: {row.get('description', '')}
Threshold Condition: {row.get('threshold_condition', '')}
Severity: {row.get('severity', 'Unknown')}
Symptoms: {row.get('symptoms', '')}
Root Cause: {row.get('root_cause', '')}
Fix Action: {row.get('fix_action', '')}
            """.strip()

            doc = DocumentClass(
                page_content=content,
                metadata={
                    "source": "industry_faults",
                    "fault_id": str(row.get("fault_id", "")),
                    "error_code": str(row.get("error_code", "")),
                    "component": str(row.get("component", "")),
                    "severity": str(row.get("severity", "")),
                    "type": "fault",
                },
            )
            documents.append(doc)

        return documents

    def _load_manual_json(self, path: str) -> List[Document]:
        """
        Load vehicle manual specifications.

        Args:
            path: Path to vehicle_manual.json

        Returns:
            List of Document objects
        """
        documents = []

        with open(path, "r") as f:
            manual = json.load(f)

        # Component specifications
        components = manual.get("components", {})
        for comp_name, specs in components.items():
            content = f"Component: {comp_name}\n"
            content += "Specifications:\n"
            for key, value in specs.items():
                content += f"  - {key}: {value}\n"

            doc = DocumentClass(
                page_content=content,
                metadata={
                    "source": "vehicle_manual",
                    "component": comp_name,
                    "type": "specification",
                },
            )
            documents.append(doc)

        # Warning thresholds
        thresholds = manual.get("warning_thresholds", {})
        if thresholds:
            content = "Warning Thresholds:\n"
            for key, value in thresholds.items():
                content += f"  - {key}: {value}\n"

            doc = DocumentClass(
                page_content=content,
                metadata={"source": "vehicle_manual", "type": "thresholds"},
            )
            documents.append(doc)

        # Driver profiles
        profiles = manual.get("driver_profiles", {})
        for profile_name, params in profiles.items():
            content = f"Driver Profile: {profile_name}\n"
            content += "Parameters:\n"
            for key, value in params.items():
                content += f"  - {key}: {value}\n"

            doc = DocumentClass(
                page_content=content,
                metadata={
                    "source": "vehicle_manual",
                    "profile": profile_name,
                    "type": "driver_profile",
                },
            )
            documents.append(doc)

        # Maintenance schedule
        maintenance = manual.get("maintenance_schedule", {})
        for interval, tasks in maintenance.items():
            content = f"Maintenance at {interval}:\n"
            for task in tasks:
                content += f"  - {task}\n"

            doc = DocumentClass(
                page_content=content,
                metadata={
                    "source": "vehicle_manual",
                    "interval": interval,
                    "type": "maintenance",
                },
            )
            documents.append(doc)

        return documents

    def load_knowledge_base(self):
        """Load existing vector database."""
        if not LANGCHAIN_AVAILABLE or not self.embeddings:
            # Silent - will build on first use
            return False

        if not os.path.exists(self.persist_dir):
            # Silent - will be created when needed
            return False

        try:
            self.vectorstore = Chroma(
                collection_name=self.COLLECTION_NAME,
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
            )
            print(f"✅ Loaded knowledge base from {self.persist_dir}")
            return True
        except Exception as e:
            print(f"❌ Error loading knowledge base: {e}")
            return False

    def retrieve_similar_faults(
        self, symptom_description: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar historical faults by symptom description.

        Args:
            symptom_description: Description of current symptoms
            k: Number of results to return

        Returns:
            List of matching fault documents with scores
        """
        if not self.vectorstore:
            return []

        try:
            results = self.vectorstore.similarity_search_with_score(
                symptom_description, k=k, filter={"type": "fault"}
            )

            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score),
                }
                for doc, score in results
            ]
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []

    def get_component_specs(self, component: str) -> Optional[Dict[str, Any]]:
        """
        Get vehicle manual specs for a component.

        Args:
            component: Component name (battery, motor, inverter, etc.)

        Returns:
            Component specifications or None
        """
        if not self.vectorstore:
            return None

        try:
            results = self.vectorstore.similarity_search(
                f"Component: {component} specifications",
                k=1,
                filter={"component": component},
            )

            if results:
                return {
                    "content": results[0].page_content,
                    "metadata": results[0].metadata,
                }
            return None
        except Exception as e:
            print(f"Error getting component specs: {e}")
            return None

    def get_maintenance_for_anomaly(
        self, anomaly_type: str, k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get relevant maintenance tasks for an anomaly type.

        Args:
            anomaly_type: Type of anomaly detected
            k: Number of results

        Returns:
            List of relevant maintenance documents
        """
        if not self.vectorstore:
            return []

        try:
            # Map anomaly types to search queries
            query_map = {
                "thermal_warning": "battery cooling temperature maintenance",
                "thermal_critical": "battery thermal runaway overheating",
                "power_anomaly": "power consumption efficiency check",
                "efficiency_degradation": "brake caliper wheel alignment efficiency",
                "mechanical_wear": "bearing wear suspension maintenance",
                "low_battery": "battery charging capacity",
            }

            query = query_map.get(anomaly_type, anomaly_type)

            results = self.vectorstore.similarity_search(query, k=k)

            return [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in results
            ]
        except Exception as e:
            print(f"Error getting maintenance: {e}")
            return []

    def semantic_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        General semantic search across all knowledge.

        Args:
            query: Natural language query
            k: Number of results

        Returns:
            List of matching documents
        """
        if not self.vectorstore:
            return []

        try:
            results = self.vectorstore.similarity_search(query, k=k)

            return [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in results
            ]
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []

    def get_faults_by_component(
        self, component: str, k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get all faults related to a specific component.

        Args:
            component: Component name
            k: Maximum results

        Returns:
            List of fault documents
        """
        if not self.vectorstore:
            return []

        try:
            results = self.vectorstore.similarity_search(
                f"{component} fault failure error", k=k, filter={"component": component}
            )

            return [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in results
            ]
        except Exception as e:
            print(f"Error getting component faults: {e}")
            return []

    def augment_training_data(
        self, telemetry_df: pd.DataFrame, anomaly_column: str = None
    ) -> pd.DataFrame:
        """
        Augment training data with RAG context.

        For each anomaly in the training data, adds relevant
        fault patterns and maintenance information.

        Args:
            telemetry_df: Training data DataFrame
            anomaly_column: Column containing anomaly types

        Returns:
            Augmented DataFrame
        """
        if not self.vectorstore:
            return telemetry_df

        # Add RAG context columns
        telemetry_df = telemetry_df.copy()
        telemetry_df["rag_fault_match"] = ""
        telemetry_df["rag_maintenance_advice"] = ""

        # For high-temperature readings, add relevant context
        if "battery_temp_c" in telemetry_df.columns:
            high_temp_mask = telemetry_df["battery_temp_c"] > 50

            if high_temp_mask.any():
                faults = self.retrieve_similar_faults(
                    "battery overheating temperature", k=1
                )
                if faults:
                    telemetry_df.loc[high_temp_mask, "rag_fault_match"] = (
                        faults[0].get("metadata", {}).get("fault_id", "")
                    )

        return telemetry_df


def build_ml_knowledge_base():
    """CLI function to build the ML knowledge base."""
    base_path = Path(__file__).parent.parent / "data" / "datasets"
    faults_path = base_path / "industry_faults.csv"
    manual_path = base_path / "vehicle_manual.json"

    print("🔧 Building ML Knowledge Base")
    print("=" * 50)

    kb = MLKnowledgeBase()
    kb.build_knowledge_base(
        faults_csv_path=str(faults_path) if faults_path.exists() else None,
        manual_json_path=str(manual_path) if manual_path.exists() else None,
    )


if __name__ == "__main__":
    import sys

    if "--build" in sys.argv:
        build_ml_knowledge_base()
    else:
        # Test queries
        print("🔍 Testing ML Knowledge Base")
        print("=" * 50)

        kb = MLKnowledgeBase()

        if kb.load_knowledge_base():
            print("\n📋 Testing similarity search for 'battery overheating'...")
            results = kb.retrieve_similar_faults(
                "battery overheating high temperature", k=3
            )
            print(f"   Found {len(results)} results")
            for r in results:
                print(
                    f"   - {r['metadata'].get('fault_id')}: {r['metadata'].get('description', 'N/A')[:50]}..."
                )

            print("\n🔧 Testing component specs for 'battery'...")
            specs = kb.get_component_specs("battery")
            if specs:
                print(f"   Found specs: {specs['content'][:100]}...")
        else:
            print("Run with --build flag first to create knowledge base")
