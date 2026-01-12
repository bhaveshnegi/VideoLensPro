import os
from pathlib import Path
import torch
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
        # Iterate through all .jpg frames
        for frame_path in sorted(frames_dir.glob("*.jpg")):
            bgr_image = cv2.imread(str(frame_path))
            if bgr_image is None:
                continue

            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb_image)

            # Detect face and extract cropped face tensor
            face_tensor = self.mtcnn(img_pil)
            if face_tensor is None:
                continue

            # Compute embedding (512-dim vector)
            with torch.no_grad():
                embedding = self.model(face_tensor.unsqueeze(0).to(self.device)).cpu().numpy()[0]
                return embedding.astype(np.float32)

        # fallback if no face detected
        return np.random.rand(512).astype(np.float32)
