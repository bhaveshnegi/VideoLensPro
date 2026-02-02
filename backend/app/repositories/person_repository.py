from datetime import datetime
import numpy as np
from bson import ObjectId

class PersonRepository:
    def __init__(self, db):
        self.collection = db["persons"]
    
    def create(self, embedding):
        result = self.collection.insert_one({
            "embedding": embedding.tolist(),
            "created_at": datetime.utcnow()
        })
        return str(result.inserted_id)
    
    def find_by_embedding(self, embedding, threshold=0.35):
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        for doc in self.collection.find({}, {"embedding": 1}):
            stored = np.array(doc["embedding"], dtype=np.float32)
            if stored.shape != embedding.shape:
                continue
                
            stored = stored / (np.linalg.norm(stored) + 1e-8)
            
            if np.dot(embedding, stored) > (1 - threshold):
                return str(doc["_id"])
        
        return None