import math
import os
import time
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any
from config import settings

class VectorDBClient:
    def __init__(self):
        # We try to load Pinecone keys. If none, we might run in memory/mock mode for local dev.
        self.api_key = os.environ.get("PINECONE_API_KEY")
        self.index_name = settings.pinecone_index_name
        self.dimension = settings.embedding_dimensions
        
        self.pc = None
        self.index = None
        
        if self.api_key and self.api_key != "your-pinecone-api-key":
            self.pc = Pinecone(api_key=self.api_key)
            self._ensure_index()
            print("Pinecone client initialized.")
        else:
            print("Warning: PINECONE_API_KEY not set. Operating in mock mode for storing embeddings.")
            self.mock_db = []

    def _ensure_index(self):
        """Creates the index if it doesn't exist."""
        if self.index_name not in [idx.name for idx in self.pc.list_indexes()]:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        deadline = time.monotonic() + 120
        while True:
            description = self.pc.describe_index(self.index_name)
            if description.dimension != self.dimension:
                raise ValueError(
                    f"Index {self.index_name!r} has dimension {description.dimension}; "
                    f"expected {self.dimension}. Set PINECONE_INDEX_NAME to a new index "
                    "to preserve the existing data during migration."
                )
            if description.status["ready"]:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Pinecone index {self.index_name!r} was not ready within 120 seconds.")
            time.sleep(1)
        self.index = self.pc.Index(self.index_name)

    def upsert(self, items: List[Dict[str, Any]]):
        """
        Takes a list of dictionaries with structure:
        {"id": str, "values": List[float], "metadata": Dict}
        """
        if self.index:
            self.index.upsert(vectors=items)
            print(f"Upserted {len(items)} items to Pinecone.")
        else:
            by_id = {item["id"]: item for item in self.mock_db}
            by_id.update({item["id"]: item for item in items})
            self.mock_db = list(by_id.values())
            print(f"Mock upserted {len(items)} items.")

    def search(self, vector: List[float], top_k: int = 5) -> List[Dict]:
        if self.index:
            response = self.index.query(vector=vector, top_k=top_k, include_metadata=True)
            return response.matches
        return self._mock_search(vector=vector, top_k=top_k)

    def search_by_type(self, vector: List[float], content_type: str, top_k: int = 5, metadata_filter: dict | None = None) -> List[Dict]:
        """Searches vectors filtered by stored content type."""
        combined_filter = {"$and": [{"type": content_type}, metadata_filter]} if metadata_filter else {"type": content_type}
        _validate_filter(combined_filter)
        if self.index:
            response = self.index.query(
                vector=vector,
                top_k=top_k,
                filter=combined_filter,
                include_metadata=True,
            )
            return response.matches
        return self._mock_search(vector=vector, top_k=top_k, metadata_filter=combined_filter)

    def delete_all(self):
        """Clears the entire index."""
        if self.index:
            try:
                self.index.delete(delete_all=True)
                print("Cleared all vectors from index.")
            except Exception as e:
                # If namespace is missing or index is already empty, ignore
                if "not found" in str(e).lower():
                    print("Index already empty or namespace not found.")
                else:
                    raise e
        else:
            self.mock_db = []
            print("Cleared mock database.")

    def _mock_search(
        self,
        vector: List[float],
        top_k: int,
        content_type: str | None = None,
        metadata_filter: dict | None = None,
    ) -> List[Dict[str, Any]]:
        candidates = self.mock_db
        if content_type:
            candidates = [
                item for item in candidates
                if item.get("metadata", {}).get("type") == content_type
            ]
        if metadata_filter:
            candidates = [item for item in candidates if _matches_filter(item.get("metadata", {}), metadata_filter)]

        scored = []
        for item in candidates:
            score = cosine_similarity(vector, item.get("values", []))
            scored.append({
                "id": item.get("id"),
                "score": score,
                "metadata": item.get("metadata", {}),
            })
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]


def _validate_filter(expression: dict) -> None:
    if not isinstance(expression, dict) or not expression:
        raise ValueError("Metadata filters must be nonempty objects.")
    operators = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "$exists"}
    for key, condition in expression.items():
        if key in {"$and", "$or"}:
            if not isinstance(condition, list) or not condition:
                raise ValueError(f"{key} requires a nonempty list of filters.")
            for child in condition:
                _validate_filter(child)
        elif key.startswith("$"):
            raise ValueError(f"Unsupported metadata filter operator: {key}")
        elif isinstance(condition, dict):
            if not condition or any(op not in operators for op in condition):
                raise ValueError("Unsupported metadata field filter.")
            for operator, operand in condition.items():
                if operator in {"$in", "$nin"} and not isinstance(operand, list):
                    raise ValueError(f"{operator} requires a list.")
                if operator == "$exists" and not isinstance(operand, bool):
                    raise ValueError("$exists requires a boolean.")
                if operator in {"$eq", "$ne"} and not isinstance(operand, (str, int, float, bool)):
                    raise ValueError(f"{operator} requires a scalar value.")
                if operator in {"$gt", "$gte", "$lt", "$lte"} and (isinstance(operand, bool) or not isinstance(operand, (int, float))):
                    raise ValueError(f"{operator} requires a number.")
                if operator in {"$in", "$nin"} and any(not isinstance(value, (str, int, float)) or isinstance(value, bool) for value in operand):
                    raise ValueError(f"{operator} requires a list of strings or numbers.")
        elif not isinstance(condition, (str, int, float, bool)):
            raise ValueError("Metadata field equality requires a scalar value.")


def _matches_filter(metadata: dict, expression: dict) -> bool:
    for key, condition in expression.items():
        if key == "$and":
            if not all(_matches_filter(metadata, child) for child in condition):
                return False
            continue
        if key == "$or":
            if not any(_matches_filter(metadata, child) for child in condition):
                return False
            continue
        exists = key in metadata
        actual = metadata.get(key)
        conditions = condition if isinstance(condition, dict) else {"$eq": condition}
        for operator, expected in conditions.items():
            values = actual if isinstance(actual, list) else [actual]
            if operator == "$exists":
                matched = exists == expected
            elif operator == "$eq":
                matched = exists and expected in values
            elif operator == "$ne":
                matched = exists and expected not in values
            elif operator == "$in":
                matched = exists and any(value in expected for value in values)
            elif operator == "$nin":
                matched = exists and all(value not in expected for value in values)
            else:
                try:
                    matched = exists and isinstance(actual, (int, float)) and not isinstance(actual, bool) and {"$gt": lambda: actual > expected, "$gte": lambda: actual >= expected,
                        "$lt": lambda: actual < expected, "$lte": lambda: actual <= expected}[operator]()
                except TypeError:
                    matched = False
            if not matched:
                return False
    return True


def cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)

if __name__ == "__main__":
    client = VectorDBClient()
