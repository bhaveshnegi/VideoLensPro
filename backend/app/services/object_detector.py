from pathlib import Path
from ultralytics import YOLO
import cv2
import uuid
import os

class ObjectDetector:
    def __init__(self):
        self.model = YOLO("models/Yolo/yolov5nu.pt")
        # Mapping between user product names and YOLO labels
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
    
    def detect_objects(self, frames_dir: Path, target_label: str = None):
        """
        Detect objects in frames. 
        If target_label is provides, focuses on that. 
        Otherwise detects all classes.
        """
        detections = []
        labels_saved = set()
        person_found = False

        for frame_path in sorted(frames_dir.glob("*.jpg")):
            results = self.model.predict(str(frame_path), imgsz=416, conf=0.25, verbose=False)[0]
            
            frame_img = None

            for box in results.boxes:
                label = results.names[int(box.cls)].lower()
                
                if label == "person":
                    person_found = True
                
                # If target_label is specified, only save that. 
                # Otherwise save all unique labels encountered.
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
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
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
                                "image_url": f"/static/detections/{filename}"
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