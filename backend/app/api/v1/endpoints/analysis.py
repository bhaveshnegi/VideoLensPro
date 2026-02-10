import tempfile
import shutil
import hashlib
import time
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, File, UploadFile, Form, Depends, Request, HTTPException

from app.dependencies import get_database
from app.core.security import get_client_info
from app.core.validation import validate_request_data, get_file_info, sanitize_filename
from app.core.logging import get_logger, log_performance, log_error
from app.core.metrics import record_video_processing_metrics, record_model_usage_metrics
from app.core.tasks import submit_video_analysis_task, get_task_status
from app.services.video_processor import get_metadata, extract_audio, extract_frames
from app.services.object_detector import ObjectDetector
from app.services.transcription import TranscriptionService
from app.services.face_recognition import FaceRecognitionService
from app.services.video_quality_service import VideoQualityService
from app.services.search_service import search_service
from app.services.vector_db import vector_db
from app.repositories.person_repository import PersonRepository
from app.repositories.upload_repository import UploadRepository
from app.core.config import settings

router = APIRouter()
logger = get_logger(__name__)

@router.post("/analyze")
async def analyze_video(
    request: Request,
    file: UploadFile = File(...),
    detection: List[str] = Form(...),
    targetType: Optional[str] = Form(None),
    target_image: Optional[UploadFile] = File(None),
    background: bool = Form(default=False),
    db = Depends(get_database)
):
    """
    Analyze video for content compliance based on frontend selections.
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    client_info = get_client_info(request)
    
    logger.info(f"Video analysis request started", extra={
        "extra_fields": {
            "request_id": request_id,
            "client_info": client_info,
            "detection": detection,
            "targetType": targetType,
            "background": background,
            "file_info": get_file_info(file)
        }
    })
    
    start_time = time.time()
    
    try:
        # Validate request data
        is_valid, validation_errors = validate_request_data(file)
        if not is_valid:
            logger.warning(f"Validation failed: {validation_errors}", extra={
                "extra_fields": {"request_id": request_id, "errors": validation_errors}
            })
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Validation failed",
                    "details": validation_errors,
                    "request_id": request_id
                }
            )
        
        # Save uploaded video file
        sanitized_video_name = sanitize_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(sanitized_video_name).suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            video_path = Path(tmp.name)
        
        # Handle target image if provided
        target_image_path = None
        if target_image:
            sanitized_img_name = sanitize_filename(target_image.filename)
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(sanitized_img_name).suffix) as tmp:
                shutil.copyfileobj(target_image.file, tmp)
                target_image_path = Path(tmp.name)
        
        try:
            # Initialize repositories
            person_repo = PersonRepository(db)
            upload_repo = UploadRepository(db)
            
            # If background processing requested
            if background and settings.ENABLE_BACKGROUND_TASKS:
                task_id = await submit_video_analysis_task({
                    "video_path": str(video_path),
                    "detection": detection,
                    "targetType": targetType,
                    "target_image_path": str(target_image_path) if target_image_path else None
                })
                
                return {
                    "task_id": task_id,
                    "status": "processing",
                    "message": "Video analysis started in background",
                    "request_id": request_id
                }
            
            # Synchronous processing
            result = await _process_video_sync(
                video_path=video_path,
                detection=detection,
                targetType=targetType,
                target_image_path=target_image_path,
                person_repo=person_repo,
                upload_repo=upload_repo,
                request_id=request_id
            )
            
            # Record metrics
            processing_time_ms = (time.time() - start_time) * 1000
            file_size_mb = file.size / (1024 * 1024) if file.size else 0
            
            record_video_processing_metrics(
                duration_ms=processing_time_ms,
                file_size_mb=file_size_mb,
                success=True  # Always successful if we reach here
            )
            
            log_performance(logger, "video_analysis", processing_time_ms, request_id, {
                "file_size_mb": file_size_mb,
                "detection": detection,
                "success": True
            })
            
            result["request_id"] = request_id
            return result
            
        finally:
            # Clean up temporary files
            try:
                if video_path.exists():
                    video_path.unlink()
                if target_image_path and target_image_path.exists():
                    target_image_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up temp files: {e}")
                
    except HTTPException:
        raise
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        log_error(logger, e, request_id)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Video analysis failed",
                "message": str(e),
                "request_id": request_id
            }
        )

async def _detect_motion(frames_dir: Path, threshold: float = 25.0) -> Dict[str, Any]:
    """
    Detect motion in video frames using frame differencing.
    
    Args:
        frames_dir: Directory containing extracted frames
        threshold: Sensitivity threshold for motion detection (lower = more sensitive)
    
    Returns:
        Dictionary with motion detection results
    """
    import cv2
    
    frames = sorted(frames_dir.glob("*.jpg"))
    if len(frames) < 2:
        return {"detected": False, "regions": []}
    
    motion_detected = False
    motion_regions = []
    
    # Sample frames (check every 5th frame for performance)
    sample_frames = frames[::5]
    
    for i in range(len(sample_frames) - 1):
        frame1 = cv2.imread(str(sample_frames[i]), cv2.IMREAD_GRAYSCALE)
        frame2 = cv2.imread(str(sample_frames[i + 1]), cv2.IMREAD_GRAYSCALE)
        
        if frame1 is None or frame2 is None:
            continue
        
        # Calculate frame difference
        diff = cv2.absdiff(frame1, frame2)
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Count non-zero pixels (motion pixels)
        motion_pixels = cv2.countNonZero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        motion_percentage = (motion_pixels / total_pixels) * 100
        
        # If more than 1% of pixels changed, consider it motion
        if motion_percentage > 1.0:
            motion_detected = True
            motion_regions.append({
                "frame_index": i * 5,
                "motion_percentage": round(motion_percentage, 2)
            })
    
    return {
        "detected": motion_detected,
        "regions": motion_regions[:10]  # Return max 10 regions to avoid large response
    }


async def _process_video_sync(
    video_path: Path,
    detection: List[str],
    targetType: Optional[str],
    target_image_path: Optional[Path],
    person_repo: PersonRepository,
    upload_repo: UploadRepository,
    request_id: str
) -> Dict[str, Any]:
    """Process video synchronously based on selected analysis types"""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        frames_dir = None
        
        # Initialize new response structure
        results = {
            "metadata": {},
            "transcript": "",
            "analysis_results": {
                "object_detection": {
                    "enabled": "objects" in detection,
                    "detections": []
                },
                "face_detection": {
                    "enabled": "faces" in detection,
                    "faces": []
                },
                "motion_analysis": {
                    "enabled": "motion" in detection,
                    "motion_detected": False,
                    "motion_regions": []
                },
                "specific_search": {
                    "enabled": "target" in detection,
                    "target_type": targetType if "target" in detection else None,
                    "matches": []
                }
            }
        }
        
        # Metadata
        metadata = get_metadata(video_path)
        results["metadata"] = metadata
        
        # Initialize services
        face_service = FaceRecognitionService()
        object_detector = ObjectDetector()
        
        # Extract frames only once if needed
        if any(x in detection for x in ["faces", "objects", "target"]):
            frames_dir = extract_frames(video_path, tmp_path / "frames")

        # 1. Face Processing
        unique_faces = []
        if frames_dir and ("faces" in detection or ("target" in detection and targetType == "face")):
            unique_faces = face_service.get_all_unique_faces(frames_dir)
            
            # Map detected faces for output
            if "faces" in detection:
                for i, face_data in enumerate(unique_faces):
                    emb = face_data["embedding"]
                    person_id = person_repo.find_by_embedding(emb)
                    if not person_id:
                        person_id = person_repo.create(emb)
                    
                    results["analysis_results"]["face_detection"]["faces"].append({
                        "label": f"Person_{person_id[:6]}",
                        "image_url": face_data["image_url"]
                    })

        # 2. Object Detection (General)
        if "objects" in detection and frames_dir:
            det_result = object_detector.detect_objects(frames_dir)
            results["analysis_results"]["object_detection"]["detections"] = det_result.get("detections", [])

        # 3. Motion Analysis
        if "motion" in detection and frames_dir:
            # Implement basic motion detection using frame differencing
            motion_detected = await _detect_motion(frames_dir)
            results["analysis_results"]["motion_analysis"]["motion_detected"] = motion_detected["detected"]
            results["analysis_results"]["motion_analysis"]["motion_regions"] = motion_detected.get("regions", [])

        # 4. Specific Search / Target Matching
        if "target" in detection and target_image_path and frames_dir:
            if targetType == "face":
                target_result = face_service.get_image_embedding(target_image_path)
                target_emb = target_result["embedding"]
                if target_emb is not None:
                    target_norm = target_emb / (np.linalg.norm(target_emb) + 1e-8)
                    
                    for face_data in unique_faces:
                        face_emb = face_data["embedding"]
                        face_norm = face_emb / (np.linalg.norm(face_emb) + 1e-8)
                        
                        if np.dot(target_norm, face_norm) > 0.65:
                            results["analysis_results"]["specific_search"]["matches"].append({
                                "label": "Target Person Found",
                                "image_url": face_data["image_url"],
                                "confidence": float(np.dot(target_norm, face_norm))
                            })
            
            elif targetType == "object":
                # For object search, detect all objects and show them as potential matches
                det_result = object_detector.detect_objects(frames_dir)
                all_detections = det_result.get("detections", [])
                # Store all detected objects as matches for object search
                results["analysis_results"]["specific_search"]["matches"] = all_detections

        # 5. Semantic Indexing (New Feature)
        if frames_dir:
            logger.info(f"Starting semantic indexing for job {request_id}")
            frame_embeddings = search_service.embed_frames(frames_dir)
            if frame_embeddings:
                f_ids = [f"{request_id}_{f['filename']}" for f in frame_embeddings]
                f_embs = [f["embedding"] for f in frame_embeddings]
                f_metas = [{
                    "job_id": request_id,
                    "filename": f["filename"],
                    "path": f["path"],
                    "video_name": metadata.get("filename", "unknown")
                } for f in frame_embeddings]
                
                vector_db.add_frames(
                    job_id=request_id,
                    frame_ids=f_ids,
                    embeddings=f_embs,
                    metadatas=f_metas
                )
                logger.info(f"Indexed {len(frame_embeddings)} frames for job {request_id}")

        # 6. Transcription
        transcription_service = TranscriptionService()
        try:
            audio_path = extract_audio(video_path, tmp_path / "audio.wav")
            transcript = transcription_service.transcribe(audio_path)
            results["transcript"] = transcript.get("text", "")
        except Exception as e:
            logger.warning(f"Transcription failed: {e}")
            results["transcript"] = ""
        
        # Save upload record (simplified without score)
        upload_repo.create({
            "detection_types": detection,
            "metadata": metadata,
            "timestamp": time.time()
        })
        
        return results

@router.get("/task/{task_id}")
async def get_task_status_endpoint(task_id: str, request: Request):
    """Get background task status"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    task_status = await get_task_status(task_id)
    if not task_status:
        raise HTTPException(status_code=404, detail={"error": "Task not found", "request_id": request_id})
    task_status["request_id"] = request_id
    return task_status

@router.get("/health")
async def health():
    return {"status": "healthy", "service": "video_analysis"}