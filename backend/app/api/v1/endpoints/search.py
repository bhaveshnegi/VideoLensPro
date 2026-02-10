from fastapi import APIRouter, File, UploadFile, Query, HTTPException
from typing import List, Dict, Any, Optional
import tempfile
import shutil
import os
from pathlib import Path

from app.services.vector_db import vector_db
from app.services.search_service import search_service
from app.services.face_recognition import FaceRecognitionService
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)
face_service = FaceRecognitionService()

@router.get("/semantic")
async def semantic_search(
    q: str = Query(..., description="Semantic search query (e.g. 'a dog in a park')"),
    limit: int = Query(5, ge=1, le=50)
):
    """
    Perform semantic search across all indexed video frames.
    """
    logger.info(f"Semantic search query: {q}")
    
    # 1. Embed query text
    query_embedding = search_service.embed_text(q)
    if not query_embedding:
        raise HTTPException(status_code=500, detail="Failed to generate query embedding")
    
    # 2. Query ChromaDB
    results = vector_db.query_frames(query_embedding, n_results=limit)
    
    # 3. Format results
    matches = []
    if results and "ids" in results and len(results["ids"][0]) > 0:
        for i in range(len(results["ids"][0])):
            matches.append({
                "id": results["ids"][0][i],
                "score": 1.0 - results["distances"][0][i],  # Convert distance to similarity
                "metadata": results["metadatas"][0][i]
            })
    
    return {
        "query": q,
        "matches": matches
    }

@router.post("/face")
async def face_search(
    file: UploadFile = File(...),
    limit: int = Query(5, ge=1, le=50)
):
    """
    Search for a person across all videos by uploading their photo.
    """
    logger.info(f"Face search with image: {file.filename}")
    
    # 1. Save temp image
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    
    try:
        # 2. Extract face embedding
        face_result = face_service.get_image_embedding(tmp_path)
        embedding = face_result.get("embedding")
        
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected in the uploaded image")
        
        # 3. Query ChromaDB
        results = vector_db.query_faces(embedding.tolist(), n_results=limit)
        
        # 4. Format results
        matches = []
        if results and "ids" in results and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                matches.append({
                    "id": results["ids"][0][i],
                    "score": 1.0 - results["distances"][0][i],
                    "metadata": results["metadatas"][0][i]
                })
        
        return {
            "filename": file.filename,
            "matches": matches
        }
        
    finally:
        if tmp_path.exists():
            os.unlink(tmp_path)

@router.get("/info")
async def search_info():
    """Get information about indexed data."""
    return {
        "faces_count": vector_db.faces_collection.count(),
        "frames_count": vector_db.frames_collection.count()
    }
