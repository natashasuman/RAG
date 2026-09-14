from typing import List, Tuple
from sentence_transformers import CrossEncoder
from app.core.config import settings


class CrossEncoderReranker:
    """Wrapper around CrossEncoder for reranking retrieved candidates."""

    def __init__(self, model_name: str = settings.RERANKER_MODEL_NAME):
        self.model = CrossEncoder(model_name)

    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Compute cross-encoder relevance scores for (query, document) pairs."""
        if not pairs:
            return []
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]
