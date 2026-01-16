import tempfile
import shutil
import hashlib
import time
import asyncio
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
                success=result.get("passed", False)
            )
            
            log_performance(logger, "video_analysis", processing_time_ms, request_id, {
                "file_size_mb": file_size_mb,
                "detection": detection,
                "success": result.get("passed", False)
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
        results = {"passed": True, "fail_reasons": []}
        
        # Metadata
        metadata = get_metadata(video_path)
        results["metadata"] = metadata
        results["detection_types"] = detection
        results["detected_objects"] = []
        
        # Face processing if requested
        person_id = None
        frames_dir = None
        results["detected_faces"] = []
        
        face_service = FaceRecognitionService()
        
        if "faces" in detection or ( "target" in detection and targetType == "face"):
            frames_dir = extract_frames(video_path, tmp_path / "frames")
            face_result = face_service.get_face_embedding(frames_dir)
            face_emb = face_result["embedding"]
            face_url = face_result["image_url"]
            
            if face_emb is not None:
                person_id = person_repo.find_by_embedding(face_emb)
                if not person_id:
                    person_id = person_repo.create(face_emb)
                
                label = f"Person_{person_id[:6]}"
                results["detected_faces"].append({
                    "label": label,
                    "image_url": face_url
                })
            
            if "faces" in detection and face_emb is None:
                results["passed"] = False
                results["fail_reasons"].append("No person detected for face analysis")
        
        results["person_id"] = person_id

        # Specific Search Matching logic
        results["matches_found"] = []
        if "target" in detection and target_image_path:
            if targetType == "face":
                target_result = face_service.get_image_embedding(target_image_path)
                target_emb = target_result["embedding"]
                if target_emb is not None:
                    # Search for this face in the video frames
                    if face_emb is not None:
                        # Normalize and compare
                        target_norm = target_emb / (np.linalg.norm(target_emb) + 1e-8)
                        face_norm = face_emb / (np.linalg.norm(face_emb) + 1e-8)
                        if np.dot(target_norm, face_norm) > 0.65: # Threshold for match
                            results["matches_found"].append({
                                "label": "Target Face Matched",
                                "image_url": target_result["image_url"] or face_url
                            })
                else:
                    results["fail_reasons"].append("Could not detect face in target image")
            
        elif targetType == "object":
                # Label matching for objects
                object_detector = ObjectDetector()
                if not frames_dir:
                    frames_dir = extract_frames(video_path, tmp_path / "frames")
                
                results["matches_found"].append({
                    "label": "Target Object Identified",
                    "image_url": None # Static for now
                })

        # Object detection if requested
        if "objects" in detection or (targetType == "object" and "target" in detection):
            if not vars().get('object_detector'):
                object_detector = ObjectDetector()
            if not frames_dir:
                frames_dir = extract_frames(video_path, tmp_path / "frames")
            
            # Use 'person' as default target if not specified
            search_label = "person"
            det_result = object_detector.detect_objects(frames_dir, search_label)
            
            # Combine found products and person_found for a better list
            results["detected_objects"] = det_result.get("detections", [])
            
            if "objects" in detection and not results["detected_objects"]:
                 results["passed"] = False
                 results["fail_reasons"].append("No relevant objects detected")

        # Motion analysis placeholder
        if "motion" in detection:
            results["motion_detected"] = True

        # Transcription
        transcription_service = TranscriptionService()
        audio_path = extract_audio(video_path, tmp_path / "audio.wav")
        transcript = transcription_service.transcribe(audio_path)
        results["transcript"] = transcript.get("text", "")

        # Scoring
        results["score"] = 100.0
        
        # Save upload record
        upload_repo.create({
            "person_id": person_id,
            "detection_types": detection,
            "score": results["score"],
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