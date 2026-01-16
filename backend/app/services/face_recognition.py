import os
from pathlib import Path
import torch
import uuid
import numpy as np
import cv2
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN

class FaceRecognitionService:
    def __init__(self):
        # Initialize directories
        self.weights_dir = Path(os.environ.get("DEEPFACE_HOME", "models")).resolve()
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load MTCNN (face detector) and Facenet model
        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        # print(f"✅ Loaded Facenet (vggface2) on {self.device.upper()}")

    def get_face_embedding(self, frames_dir: Path):
        """Extract embedding and image path from the first face found in a directory of frames."""
        for frame_path in sorted(frames_dir.glob("*.jpg")):
            result = self.get_image_embedding(frame_path, save_crop=True)
            if result["embedding"] is not None:
                return result

        return {"embedding": None, "image_url": None}

    def get_all_unique_faces(self, frames_dir: Path, threshold: float = 0.6):
        """Extract all unique faces found in a directory of frames."""
        unique_faces = []
        known_embeddings = []

        for frame_path in sorted(frames_dir.glob("*.jpg")):
            result = self.get_image_embedding(frame_path, save_crop=True)
            emb = result["embedding"]
            
            if emb is not None:
                is_unique = True
                # Normalize current embedding
                emb_norm = emb / (np.linalg.norm(emb) + 1e-8)

                for known_emb in known_embeddings:
                    # Normalize known embedding
                    known_norm = known_emb / (np.linalg.norm(known_emb) + 1e-8)
                    
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
        """Extract embedding from a single image file, optionally saving a crop."""
        bgr_image = cv2.imread(str(image_path))
        if bgr_image is None:
            return {"embedding": None, "image_url": None}

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_image)

        image_url = None
        if save_crop:
            boxes, _ = self.mtcnn.detect(img_pil)
            if boxes is not None:
                x1, y1, x2, y2 = map(int, boxes[0])
                h, w = bgr_image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                crop = bgr_image[y1:y2, x1:x2]
                if crop.size > 0:
                    filename = f"face_{uuid.uuid4().hex[:8]}.jpg"
                    save_path = f"static/detections/{filename}"
                    os.makedirs("static/detections", exist_ok=True)
                    cv2.imwrite(save_path, crop)
                    image_url = f"/static/detections/{filename}"

        # Get embedding
        face_tensor = self.mtcnn(img_pil)
        if face_tensor is None:
            return {"embedding": None, "image_url": image_url}

        with torch.no_grad():
            embedding = self.model(face_tensor.unsqueeze(0).to(self.device)).cpu().numpy()[0]
            return {
                "embedding": embedding.astype(np.float32),
                "image_url": image_url
            }
