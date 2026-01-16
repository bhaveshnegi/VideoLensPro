from datetime import datetime
from bson import ObjectId
from typing import Optional, Dict, Any, List

from app.core.logging import get_logger

logger = get_logger(__name__)


class UploadRepository:
    def __init__(self, db):
        self.collection = db["uploads"]
    
    def find_by_person_and_product(self, person_id, product_key):
        return self.collection.find_one({
            "person_id": person_id,
            "product_key": product_key
        })
    
    def create(self, upload_data):
        result = self.collection.insert_one({
            **upload_data,
            "timestamp": datetime.utcnow()
        })
        return str(result.inserted_id)
