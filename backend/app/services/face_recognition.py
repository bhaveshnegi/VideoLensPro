import os
import uuid
import numpy as np
import cv2
import torch

from pathlib import Path
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN

from app.core.logging import get_logger
from app.services.vector_db import vector_db

logger = get_logger(__name__)


class FaceRecognitionService:
    def __init__(self):
        """
        Local FaceNet pipeline:
        - MTCNN face detector
        - InceptionResnetV1 (VGGFace2 pretrained)
        - 512-dim embeddings
        """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        logger.info(f"✅ FaceNet loaded on {self.device.upper()}")

    # ---------------------------------------------------------

    def get_face_embedding(self, frames_dir: Path):
        """Return first detected face embedding from frames."""
        for frame_path in sorted(frames_dir.glob("*.jpg")):
            result = self.get_image_embedding(frame_path, save_crop=True)
            if result["embedding"] is not None:
                return result

        return {"embedding": None, "image_url": None}

    # ---------------------------------------------------------

    def get_all_unique_faces(self, frames_dir: Path, threshold: float = 0.7):
        """
        Extract unique faces using cosine similarity.
        FaceNet similarity threshold ≈ 0.6–0.8
        """
        unique_faces = []
        known_embeddings = []

        frame_paths = sorted(frames_dir.glob("*.jpg"))

        # Sampling for speed
        if len(frame_paths) > 30:
            frame_paths = frame_paths[::15]
        elif len(frame_paths) > 10:
            frame_paths = frame_paths[::5]

        for frame_path in frame_paths:
            result = self.get_image_embedding(frame_path, save_crop=True)
            emb = result["embedding"]

            if emb is None:
                continue

            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)

            is_unique = True
            for known in known_embeddings:
                known_norm = known / (np.linalg.norm(known) + 1e-8)
                if np.dot(emb_norm, known_norm) > threshold:
                    is_unique = False
                    break

            if is_unique:
                face_id = f"face_{uuid.uuid4().hex[:8]}"

                vector_db.add_face(
                    face_id=face_id,
                    embedding=emb_norm.tolist(),
                    metadata={
                        "image_url": result["image_url"],
                    },
                )

                known_embeddings.append(emb)
                unique_faces.append(
                    {
                        "face_id": face_id,
                        "embedding": emb,
                        "image_url": result["image_url"],
                    }
                )

        return unique_faces

    # ---------------------------------------------------------

    def get_image_embedding(self, image_path: Path, save_crop: bool = False):
        """Detect face and compute FaceNet embedding."""
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            return {"embedding": None, "image_url": None}

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)

        # Detect and align face
        face_tensor = self.mtcnn(img_pil)

        if face_tensor is None:
            return {"embedding": None, "image_url": None}

        # Compute embedding
        with torch.no_grad():
            embedding = (
                self.model(face_tensor.unsqueeze(0).to(self.device))
                .cpu()
                .numpy()[0]
                .astype(np.float32)
            )

        image_url = None

        # Save cropped face for UI
        if save_crop:
            # MTCNN returns aligned 160x160 tensor → convert back to image
            crop = face_tensor.permute(1, 2, 0).cpu().numpy()
            crop = (crop * 255).astype(np.uint8)

            filename = f"face_{uuid.uuid4().hex[:8]}.jpg"
            save_path = Path("static/detections") / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(save_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            image_url = f"/static/detections/{filename}"

        return {"embedding": embedding, "image_url": image_url}
