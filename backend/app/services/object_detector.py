from pathlib import Path
from ultralytics import YOLO

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
    
    def detect_objects(self, frames_dir: Path, product_name: str):
        product_name_lower = product_name.lower()
        expected_labels = self._get_expected_labels(product_name_lower)

        person_found = False
        product_found = False
        detected_products = []

        for frame in sorted(frames_dir.glob("*.jpg")):
            results = self.model.predict(frame, imgsz=416, conf=0.25, verbose=False)[0]

            for box in results.boxes:
                label = results.names[int(box.cls)].lower()

                if label == "person":
                    person_found = True

                if label in expected_labels:
                    product_found = True
                    if label not in detected_products:
                        detected_products.append(label)

        return {
            "person_found": person_found,
            "product_found": product_found,
            "products": detected_products,
            "expected_labels": expected_labels
        }

    def _get_expected_labels(self, product_name_lower: str):
        for key, values in self.product_map.items():
            if key in product_name_lower:
                return values
        return []  # Unknown product type