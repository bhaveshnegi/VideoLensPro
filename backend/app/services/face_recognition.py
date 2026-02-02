import requests
import time
import io
import uuid
import os
import numpy as np
import cv2
from pathlib import Path
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class FaceRecognitionService:
    def __init__(self):
        # Use local lightweight Haar Cascades for detection (fast, low RAM)
        # Use Cloud API for high-fidelity embedding extraction
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Fix suffix .xml
        self.embed_url = "https://router.huggingface.co/hf-inference/models/google/vit-base-patch16-224"
        self.headers = {
            "Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}",
            "Content-Type": "image/jpeg"
        }

    def _query_api(self, url, data, is_json=False):
        for attempt in range(3):
            try:
                if is_json:
                    response = requests.post(url, headers=self.headers, json=data, timeout=30)
                else:
                    response = requests.post(url, headers=self.headers, data=data, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 503: # Model loading
                    time.sleep(5)
                    continue
                else:
                    logger.error(f"HF API Error ({url}): {response.status_code} - {response.text}")
                    return None
            except Exception as e:
                logger.error(f"HF API request failed ({url}): {e}")
                time.sleep(2)
        return None

    def get_face_embedding(self, frames_dir: Path):
        """Extract embedding and image path from the first face found in a directory of frames."""
        for frame_path in sorted(frames_dir.glob("*.jpg")):
            result = self.get_image_embedding(frame_path, save_crop=True)
            if result["embedding"] is not None:
                return result
        return {"embedding": None, "image_url": None}

    def get_all_unique_faces(self, frames_dir: Path, threshold: float = 0.75):
        """Extract all unique faces found in a directory of frames using HF APIs."""
        unique_faces = []
        known_embeddings = []

        # Sampling frames to stay within API limits
        frame_paths = sorted(frames_dir.glob("*.jpg"))
        if len(frame_paths) > 30:
            frame_paths = frame_paths[::15]
        elif len(frame_paths) > 10:
            frame_paths = frame_paths[::5]

        for frame_path in frame_paths:
            result = self.get_image_embedding(frame_path, save_crop=True)
            emb = result["embedding"]
            
            if emb is not None:
                is_unique = True
                # Normalize current embedding
                emb_norm = emb / (np.linalg.norm(emb) + 1e-8)

                for known_emb in known_embeddings:
                    # Normalize known embedding
                    known_norm = known_emb / (np.linalg.norm(known_emb) + 1e-8)
                    # CLIP embeddings tend to have higher cosine similarity base, so we use a higher threshold (0.75-0.85)
                    if np.dot(emb_norm, known_norm) > threshold:
                        is_unique = False
                        break
                
                if is_unique:
                    known_embeddings.append(emb)
                    unique_faces.append({
                        "embedding": emb,
                        "image_url": result["image_url"]
                    })

        return unique_faces

    def get_image_embedding(self, image_path: Path, save_crop: bool = False):
        """Extract embedding from a single image file using Local Detection + HF Feature Extraction."""
        bgr_image = cv2.imread(str(image_path))
        if bgr_image is None:
            return {"embedding": None, "image_url": None}

        # 1. Detect Face Locally
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return {"embedding": None, "image_url": None}

        # Take the first face
        (x, y, w_face, h_face) = faces[0]
        h_img, w_img = bgr_image.shape[:2]
        
        # Add slight padding
        pad = 20
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(w_img, x+w_face+pad), min(h_img, y+h_face+pad)
        
        face_crop = bgr_image[y1:y2, x1:x2]
        if face_crop.size == 0:
            return {"embedding": None, "image_url": None}

        # 3. Save crop if requested
        image_url = None
        if save_crop:
            filename = f"face_{uuid.uuid4().hex[:8]}.jpg"
            save_path = f"static/detections/{filename}"
            os.makedirs("static/detections", exist_ok=True)
            cv2.imwrite(save_path, face_crop)
            image_url = f"/static/detections/{filename}"

        # 4. Get Embedding via HF Feature Extraction (Semantic Signature)
        success, buffer = cv2.imencode(".jpg", face_crop)
        if not success:
            return {"embedding": None, "image_url": image_url}
        
        crop_bytes = buffer.tobytes()
        api_result = self._query_api(self.embed_url, crop_bytes)
        
        if api_result and isinstance(api_result, list) and len(api_result) > 0:
            # If the API returns classification dicts (label/score), extract scores as a vector
            if isinstance(api_result[0], dict):
                embedding = np.array([r.get("score", 0) for r in api_result], dtype=np.float32)
            else:
                embedding = np.array(api_result, dtype=np.float32)
                
            return {
                "embedding": embedding,
                "image_url": image_url
            }

        return {"embedding": None, "image_url": image_url}
