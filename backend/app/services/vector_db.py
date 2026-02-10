import chromadb
from chromadb.config import Settings
import logging
from typing import List, Dict, Any, Optional
import os
from app.core.config import settings

logger = logging.getLogger(__name__)

class VectorDBService:
    def __init__(self):
        # In a Docker environment, we connect to the 'chromadb' service
        # In local dev, we might use localhost or a local path
        host = settings.CHROMA_HOST
        port = settings.CHROMA_PORT
        
        try:
            self.client = chromadb.HttpClient(host=host, port=port)
            logger.info(f"Connected to ChromaDB at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            # Fallback to ephemeral client for testing if server is down
            self.client = chromadb.EphemeralClient()
            logger.warning("Using EphemeralClient as fallback")

        # Initialize collections
        self.faces_collection = self.client.get_or_create_collection(
            name="faces",
            metadata={"hnsw:space": "cosine"}
        )
        self.frames_collection = self.client.get_or_create_collection(
            name="video_frames",
            metadata={"hnsw:space": "cosine"}
        )

    def add_face(self, face_id: str, embedding: List[float], metadata: Dict[str, Any]):
        """Add a face embedding to the vector database."""
        try:
            self.faces_collection.add(
                ids=[face_id],
                embeddings=[embedding],
                metadatas=[metadata]
            )
        except Exception as e:
            logger.error(f"Error adding face to ChromaDB: {e}")

    def query_faces(self, embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        """Search for similar faces."""
        try:
            return self.faces_collection.query(
                query_embeddings=[embedding],
                n_results=n_results
            )
        except Exception as e:
            logger.error(f"Error querying faces in ChromaDB: {e}")
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}

    def add_frames(self, job_id: str, frame_ids: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        """Add video frame embeddings for semantic search."""
        try:
            self.frames_collection.add(
                ids=frame_ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
        except Exception as e:
            logger.error(f"Error adding frames to ChromaDB: {e}")

    def query_frames(self, query_embedding: List[float], n_results: int = 10) -> Dict[str, Any]:
        """Search for frames by semantic meaning (text query embedding)."""
        try:
            return self.frames_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
        except Exception as e:
            logger.error(f"Error querying frames in ChromaDB: {e}")
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}

    def delete_job_data(self, job_id: str):
        """Clean up data associated with a specific job."""
        try:
            # Note: ChromaDB doesn't have a direct 'delete by metadata' in all versions
            # We use where clause in delet
            self.frames_collection.delete(where={"job_id": job_id})
            # Faces might be multi-job, so we might not want to delete them by job_id 
            # unless specifically requested or if they have a job_id tag.
        except Exception as e:
            logger.error(f"Error deleting job data from ChromaDB: {e}")

# Global instance
vector_db = VectorDBService()
