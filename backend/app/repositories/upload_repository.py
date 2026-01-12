from datetime import datetime
from bson import ObjectId
from typing import Optional, Dict, Any, List

from app.core.logging import get_logger

logger = get_logger(__name__)


class UploadRepository:
    def __init__(self, db):
        self.collection = db["uploads"]
    
    def find_by_hash(self, video_hash):
        """
        Find upload by legacy video_hash field (v1 format).
        Kept for backward compatibility.
        """
        return self.collection.find_one({"video_hash": video_hash})
    
    def find_by_file_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Find upload by exact file hash (v2 format).
        Fast check for exact duplicate files.
        """
        return self.collection.find_one({"video_file_hash": file_hash})
    
    def find_by_content_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """
        Find upload by content hash (v2 format).
        Detects perceptually similar videos.
        """
        return self.collection.find_one({"video_content_hash": content_hash})
    
    def find_similar_videos(
        self, 
        content_hash: str, 
        similarity_threshold: float = 90.0
    ) -> List[Dict[str, Any]]:
        """
        Find videos with similar content hashes.
        Uses Hamming distance for similarity matching.
        
        Note: This is a simple implementation. For better performance with
        large datasets, consider using specialized similarity search 
        (e.g., LSH, annoy, faiss).
        
        Args:
            content_hash: Content hash to compare against
            similarity_threshold: Minimum similarity percentage (0-100)
            
        Returns:
            List of similar video records
        """
        # Get all uploads with content hashes
        all_uploads = self.collection.find({
            "video_content_hash": {"$exists": True}
        })
        
        similar = []
        
        for upload in all_uploads:
            existing_hash = upload.get("video_content_hash")
            if not existing_hash:
                continue
            
            # Calculate similarity
            similarity = self._calculate_similarity(content_hash, existing_hash)
            
            if similarity >= similarity_threshold:
                upload["similarity_score"] = similarity
                similar.append(upload)
        
        # Sort by similarity descending
        similar.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        
        return similar
    
    def is_duplicate(
        self,
        file_hash: str,
        content_hash: str,
        similarity_threshold: float = 90.0
    ) -> tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Check if video is a duplicate using multi-layer approach.
        
        Args:
            file_hash: SHA256 hash of file bytes
            content_hash: Perceptual content hash
            similarity_threshold: Similarity threshold for content matching
            
        Returns:
            Tuple of (is_duplicate, match_type, matched_record)
            match_type can be: "exact_file", "exact_content", "similar_content", or None
        """
        # First check: Exact file match (fastest, most strict)
        exact_file = self.find_by_file_hash(file_hash)
        if exact_file:
            logger.info(f"Exact file match found: {exact_file.get('_id')}")
            return True, "exact_file", exact_file
        
        # Second check: Legacy hash support (v1 compatibility)
        legacy = self.find_by_hash(file_hash)
        if legacy:
            logger.info(f"Legacy hash match found: {legacy.get('_id')}")
            return True, "exact_file", legacy
        
        # Third check: Exact content match
        exact_content = self.find_by_content_hash(content_hash)
        if exact_content:
            logger.info(f"Exact content match found: {exact_content.get('_id')}")
            return True, "exact_content", exact_content
        
        # Fourth check: Similar content match (perceptual)
        similar = self.find_similar_videos(content_hash, similarity_threshold)
        if similar:
            best_match = similar[0]
            logger.info(
                f"Similar content match found: {best_match.get('_id')} "
                f"(similarity: {best_match.get('similarity_score', 0):.2f}%)"
            )
            return True, "similar_content", best_match
        
        return False, None, None
    
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
    
    def _calculate_similarity(self, hash1: str, hash2: str) -> float:
        """
        Calculate similarity percentage between two hashes.
        
        Args:
            hash1: First hash string
            hash2: Second hash string
            
        Returns:
            Similarity percentage (0-100)
        """
        if not hash1 or not hash2:
            return 0.0
        
        if len(hash1) != len(hash2):
            return 0.0
        
        # Calculate Hamming distance
        distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        max_distance = len(hash1)
        
        if max_distance == 0:
            return 100.0
        
        similarity = (1 - distance / max_distance) * 100
        return round(similarity, 2)