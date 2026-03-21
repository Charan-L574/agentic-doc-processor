from __future__ import annotations

from typing import Any, Dict, List

from services.vector_store_service import VectorStoreService
from utils.faiss_manager import get_faiss_index


class FAISSVectorStoreAdapter(VectorStoreService):
    """Adapter over existing FAISS manager implementation."""

    def __init__(self):
        self.index = get_faiss_index()

    def index(self, texts: List[str], metadata: List[Dict[str, Any]]) -> None:
        self.index.add_documents(texts=texts, metadata=metadata)
        self.index.save()

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        return self.index.search(query=query, k=k)

    def refresh(self) -> None:
        self.index._load_or_create_index()
