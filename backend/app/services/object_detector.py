import requests
import io
import time
from pathlib import Path
import cv2
import uuid
import os
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class ObjectDetector:
    def __init__(self):
        self.api_url = "https://router.huggingface.co/hf-inference/models/facebook/detr-resnet-50"
        self.headers = {
            "Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}",
            "Content-Type": "image/jpeg"
        }
        # Mapping between user product names and YOLO labels (still useful for filtering)
        self.product_map = {
            "Smartphone": ["cell phone", "mobile phone", "phone"],
            "smartphone": ["cell phone", "mobile phone", "phone"],
            "mobile": ["cell phone", "mobile phone", "phone"],
            "smartwatch": ["watch"],
            "Smartlaptop": ["laptop", "notebook"],
            "smartlaptop": ["laptop", "notebook"],
            "SmartTV": ["tv", "monitor", "television"],
            "smartTV": ["tv", "monitor", "television"],
            "tablet": ["tablet", "ipad"],
            "camera": ["camera"]
        }
    
    def _query(self, frame_path):
        with open(frame_path, "rb") as f:
            data = f.read()
        
        # Simple retry logic for the inference API (it might be loading)
        for attempt in range(3):
            try:
                response = requests.post(self.api_url, headers=self.headers, data=data, timeout=30)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 503:  # model loading
                    time.sleep(5)
                    continue
                else:
                    logger.error(f"HF API Error: {response.status_code} - {response.text}")
                    return []
            except Exception as e:
                logger.error(f"HF API request failed: {e}")
                time.sleep(2)
        return []

    def detect_objects(self, frames_dir: Path, target_label: str = None):
        """
        Detect objects in frames using Hugging Face Inference API. 
        """
        detections = []
        labels_saved = set()
        person_found = False

        # Limit frame processing to avoid hitting free tier limits too fast
        frame_paths = sorted(frames_dir.glob("*.jpg"))
        # Sample frames (every 10th instead of every single if many)
        if len(frame_paths) > 20:
            frame_paths = frame_paths[::10]

        for frame_path in frame_paths:
            results = self._query(str(frame_path))
            
            if not results or not isinstance(results, list):
                continue

            frame_img = None

            for result in results:
                label = result.get("label", "").lower()
                score = result.get("score", 0)
                box = result.get("box", {})

                if score < 0.25:
                    continue
                
                if label == "person":
                    person_found = True
                
                should_save = False
                if target_label:
                    if label == target_label or (target_label in self.product_map and label in self.product_map[target_label]):
                        should_save = label not in labels_saved
                else:
                    should_save = label not in labels_saved

                if should_save:
                    if frame_img is None:
                        frame_img = cv2.imread(str(frame_path))
                    
                    if frame_img is not None:
                        # HF box format: {'xmin': ..., 'ymin': ..., 'xmax': ..., 'ymax': ...}
                        x1 = int(box.get('xmin', 0))
                        y1 = int(box.get('ymin', 0))
                        x2 = int(box.get('xmax', 0))
                        y2 = int(box.get('ymax', 0))
                        
                        # Add padding
                        h, w = frame_img.shape[:2]
                        pad = 10
                        x1, y1 = max(0, x1-pad), max(0, y1-pad)
                        x2, y2 = min(w, x2+pad), min(h, y2+pad)

                        crop = frame_img[y1:y2, x1:x2]
                        if crop.size > 0:
                            filename = f"obj_{uuid.uuid4().hex[:8]}_{label}.jpg"
                            save_path = f"static/detections/{filename}"
                            os.makedirs("static/detections", exist_ok=True)
                            cv2.imwrite(save_path, crop)
                            detections.append({
                                "label": label.capitalize(),
                                "image_url": f"/static/detections/{filename}",
                                "confidence": round(score, 3)
                            })
                            labels_saved.add(label)

        return {
            "person_found": person_found,
            "detections": detections
        }

    def _get_expected_labels(self, product_name_lower: str):
        for key, values in self.product_map.items():
            if key in product_name_lower:
                return values
        return []  # Unknown product type