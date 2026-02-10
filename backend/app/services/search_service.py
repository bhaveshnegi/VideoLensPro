import requests
import time
import io
import logging
from typing import List, Union, Dict, Any, Optional
import numpy as np
from pathlib import Path
from app.core.config import settings

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32"):
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {
            "Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"
        }
        logger.info(f"Initialized SearchService with HF API: {model_name}")

    def _query_api(self, payload: Any, is_json: bool = True) -> Optional[List[float]]:
        """Generic helper to query HF Inference API."""
        for attempt in range(3):
            try:
                if is_json:
                    response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
                else:
                    response = requests.post(self.api_url, headers=self.headers, data=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    # For sentence-transformers models via API, it often returns a list or a list of lists
                    if isinstance(result, list):
                        if len(result) > 0 and isinstance(result[0], list):
                            return result[0]
                        return result
                    return result
                elif response.status_code == 503: # Model loading
                    logger.warning(f"HF Model loading, waiting 5s... (Attempt {attempt+1})")
                    time.sleep(5)
                    continue
                else:
                    logger.error(f"HF API Error ({self.api_url}): {response.status_code} - {response.text}")
                    return None
            except Exception as e:
                logger.error(f"HF API request failed: {e}")
                time.sleep(2)
        return None

    def embed_text(self, text: str) -> List[float]:
        """Convert a text query into a vector via HF API."""
        payload = {"inputs": text}
        result = self._query_api(payload)
        return result if result else []

    def embed_image(self, image_path: Union[str, Path]) -> List[float]:
        """Convert an image into a vector via HF API."""
        try:
            with open(str(image_path), "rb") as f:
                img_data = f.read()
            
            # For CLIP models supporting image inputs directly
            result = self._query_api(img_data, is_json=False)
            return result if result else []
        except Exception as e:
            logger.error(f"Error reading image for embedding {image_path}: {e}")
            return []

    def embed_frames(self, frames_dir: Path, sample_rate: int = 15) -> List[dict]:
        """
        Embed multiple frames from a directory via HF API.
        """
        results = []
        frame_paths = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
        sampled_paths = frame_paths[::sample_rate]
        
        logger.info(f"Embedding {len(sampled_paths)} frames from {frames_dir} via HF API")
        
        for path in sampled_paths:
            embedding = self.embed_image(path)
            if embedding:
                results.append({
                    "path": str(path),
                    "filename": path.name,
                    "embedding": embedding
                })
        
        return results

# Global instance
search_service = SearchService()
