from __future__ import annotations

from typing import Any, Dict, List

from services.vector_store_service import VectorStoreService


class OpenSearchVectorStoreAdapter(VectorStoreService):
    """Placeholder OpenSearch adapter. Concrete indexing/query wiring added in Phase 2."""

    def index(self, texts: List[str], metadata: List[Dict[str, Any]]) -> None:
        raise NotImplementedError("OpenSearch indexing implementation is pending")

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        raise NotImplementedError("OpenSearch search implementation is pending")

    def refresh(self) -> None:
        return None
