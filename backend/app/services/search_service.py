from sentence_transformers import SentenceTransformer
from typing import List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SearchService:
    """
    Option B FINAL:
    - Local MiniLM embeddings
    - Text semantic search ONLY
    - No HF router dependency
    """

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info(" MiniLM semantic search model loaded")

    # ---------------------------------------------------------

    def embed_text(self, text: str) -> List[float]:
        """Convert transcript / labels into 384-dim embedding."""
        return self.model.encode(text).tolist()

    # ---------------------------------------------------------

    def embed_frames(self, frames_dir: Path, sample_rate: int = 15):
        """
        Not used in Option B.
        Kept only to avoid breaking imports.
        """
        return []


# Global instance
search_service = SearchService()
