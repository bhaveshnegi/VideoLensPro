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
        frames_dir = None
        results = {
            "passed": True, 
            "fail_reasons": [],
            "detection_types": detection,
            "detected_objects": [],
            "detected_faces": [],
            "matches_found": []
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
        if frames_dir and ("faces" in detection or ( "target" in detection and targetType == "face")):
            unique_faces = face_service.get_all_unique_faces(frames_dir)
            
            # Map detected faces for output
            for i, face_data in enumerate(unique_faces):
                emb = face_data["embedding"]
                person_id = person_repo.find_by_embedding(emb)
                if not person_id:
                    person_id = person_repo.create(emb)
                
                results["detected_faces"].append({
                    "label": f"Person_{person_id[:6]}",
                    "image_url": face_data["image_url"]
                })

            if "faces" in detection and not results["detected_faces"]:
                results["passed"] = False
                results["fail_reasons"].append("No person detected for face analysis")

        # 2. Specific Search / Target Matching
        if "target" in detection and target_image_path and frames_dir:
            if targetType == "face":
                target_result = face_service.get_image_embedding(target_image_path)
                target_emb = target_result["embedding"]
                if target_emb is not None:
                    target_norm = target_emb / (np.linalg.norm(target_emb) + 1e-8)
                    
                    found_match = False
                    for face_data in unique_faces:
                        face_emb = face_data["embedding"]
                        face_norm = face_emb / (np.linalg.norm(face_emb) + 1e-8)
                        
                        if np.dot(target_norm, face_norm) > 0.65:
                            results["matches_found"].append({
                                "label": "Target Person Found",
                                "image_url": face_data["image_url"]
                            })
                            found_match = True
                    
                    if not found_match:
                        results["fail_reasons"].append("Target person not found in video")
                else:
                    results["fail_reasons"].append("Could not detect face in target image")
            
            elif targetType == "object":
                # For object search, we look for matches from all detected objects
                # For now, we use label-based matching or if no target_image, we use "person" as default
                # But here we assume we detect ALL objects and filter
                det_result = object_detector.detect_objects(frames_dir)
                all_detections = det_result.get("detections", [])
                
                # If we have all detections, we might want to filter by what's in the target image
                # For simplicity, if target_image is provided, we'd need an object classifier for it.
                # Let's assume the user just wants to find "Object" in video for now or specific tags.
                # We'll just show what was found if 'objects' is selected.
                pass

        # 3. Object Detection (General)
        if "objects" in detection and frames_dir:
            det_result = object_detector.detect_objects(frames_dir)
            results["detected_objects"] = det_result.get("detections", [])
            
            if not results["detected_objects"]:
                 results["passed"] = False
                 results["fail_reasons"].append("No relevant objects detected")

        # 4. Motion analysis placeholder
        if "motion" in detection:
            results["motion_detected"] = True

        # 5. Transcription
        transcription_service = TranscriptionService()
        try:
            audio_path = extract_audio(video_path, tmp_path / "audio.wav")
            transcript = transcription_service.transcribe(audio_path)
            results["transcript"] = transcript.get("text", "")
        except Exception as e:
            logger.warning(f"Transcription failed: {e}")
            results["transcript"] = ""

        # Scoring
        results["score"] = 95.0 if results["passed"] else 45.0
        
        # Save upload record
        upload_repo.create({
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