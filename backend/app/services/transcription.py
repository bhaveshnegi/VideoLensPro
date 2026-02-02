import requests
import time
from pathlib import Path
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class TranscriptionService:
    def __init__(self):
        self.transcribe_url = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3"
        self.toxicity_url = "https://router.huggingface.co/hf-inference/models/unitary/toxic-bert"
        self.auth_header = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"}

    def _query_api(self, url, data, is_json=False):
        for attempt in range(3):
            try:
                headers = self.auth_header.copy()
                if is_json:
                    headers["Content-Type"] = "application/json"
                    response = requests.post(url, headers=headers, json=data, timeout=60)
                else:
                    headers["Content-Type"] = "audio/wav"  # Whisper prefers wav/flac
                    response = requests.post(url, headers=headers, data=data, timeout=60)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 503: # Model loading
                    time.sleep(10)
                    continue
                else:
                    logger.error(f"HF API Error ({url}): {response.status_code} - {response.text}")
                    return None
            except Exception as e:
                logger.error(f"HF API request failed ({url}): {e}")
                time.sleep(2)
        return None

    def transcribe(self, audio_path: Path):
        """
        Transcribe audio using Hugging Face Whisper and detect toxicity.
        """
        logger.info(f"Transcribing audio file: {audio_path}")
        
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        # 1. Transcribe
        transcription_result = self._query_api(self.transcribe_url, audio_data)
        text = ""
        if transcription_result and isinstance(transcription_result, dict):
            text = transcription_result.get("text", "").strip()
        
        if not text:
            return {
                "text": "",
                "toxicity": None,
                "language": "unknown",
                "is_toxic": False
            }

        # 2. Toxicity Check
        toxicity_results = self._query_api(self.toxicity_url, {"inputs": text}, is_json=True)
        
        toxic_labels = []
        if toxicity_results and isinstance(toxicity_results, list) and len(toxicity_results) > 0:
            # toxic-bert returns a list of lists of dicts: [[{'label': 'toxic', 'score': ...}, ...]]
            inner_results = toxicity_results[0]
            toxic_labels = [r for r in inner_results if r["score"] > 0.5 and r["label"] != "non-toxic"]

        return {
            "text": text,
            "language": "en", # Whisper large usually detects, but free API response might be simplified
            "toxicity": toxic_labels,
            "is_toxic": len(toxic_labels) > 0,
            "confidence": {
                r["label"]: round(r["score"], 3)
                for r in toxic_labels
            }
        }
