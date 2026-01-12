import tempfile
import shutil
import hashlib
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import APIRouter, File, UploadFile, Form, Depends, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from app.dependencies import get_database
from app.core.security import verify_service_access, get_client_info
from app.core.validation import validate_request_data, get_file_info, sanitize_filename
from app.core.logging import get_logger, log_performance, log_error
from app.core.metrics import record_video_processing_metrics, record_model_usage_metrics
from app.core.config import settings
from app.core.video_hash_config import video_hash_config
from app.core.tasks import submit_video_analysis_task, get_task_status
from app.services.video_hasher import VideoHasher
from app.services.video_processor import get_metadata, extract_audio, extract_frames
from app.services.object_detector import ObjectDetector
from app.services.transcription import TranscriptionService
from app.services.face_recognition import FaceRecognitionService
from app.services.video_quality_service import VideoQualityService
from app.repositories.person_repository import PersonRepository
from app.repositories.upload_repository import UploadRepository

router = APIRouter()
logger = get_logger(__name__)

def compute_hash(file_path: Path):
    """Compute SHA256 hash of file"""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

@router.post("/analyze")
async def analyze_video(
    request: Request,
    file: UploadFile = File(...),
    model_id: str = Form(...),
    product_name: str = Form(...),
    background: bool = Form(default=False),
    db = Depends(get_database),
    _: bool = Depends(verify_service_access)
):
    """
    Analyze video for content compliance
    
    - **file**: Video file to analyze (MP4, AVI, MOV, MKV, WebM)
    - **model_id**: Model identifier for tracking
    - **background**: Process in background (returns task ID)
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    client_info = get_client_info(request)
    
    logger.info(f"Video analysis request started", extra={
        "extra_fields": {
            "request_id": request_id,
            "client_info": client_info,
            "model_id": model_id,
            "background": background,
            "file_info": get_file_info(file)
        }
    })
    
    start_time = time.time()
    
    try:
        # Validate request data
        is_valid, validation_errors = validate_request_data(file, model_id)
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
        
        # Save uploaded file
        sanitized_filename = sanitize_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(sanitized_filename).suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            video_path = Path(tmp.name)
        
        try:
            # Initialize repositories and services
            person_repo = PersonRepository(db)
            upload_repo = UploadRepository(db)
            
            # Initialize video hasher
            hasher = VideoHasher(
                frame_interval_sec=video_hash_config.HASH_FRAME_INTERVAL_SEC,
                hash_size=video_hash_config.HASH_SIZE,
                max_frames=video_hash_config.HASH_MAX_FRAMES
            )
            
            # Compute video hashes (multi-layer approach)
            hash_start = time.time()
            
            if video_hash_config.ENABLE_CONTENT_HASH:
                video_hashes = hasher.compute_all_hashes(video_path)
                file_hash = video_hashes["video_file_hash"]
                content_hash = video_hashes["video_content_hash"]
            else:
                # Fallback to simple file hash if content hash disabled
                file_hash = hasher.compute_file_hash(video_path)
                content_hash = file_hash
                video_hashes = {
                    "video_file_hash": file_hash,
                    "video_content_hash": content_hash,
                    "hash_version": "v1"
                }
            
            hash_time = (time.time() - hash_start) * 1000
            log_performance(logger, "video_hashing", hash_time, request_id)
            
            # Check for duplicate videos using multi-layer detection
            is_dup, match_type, matched_record = upload_repo.is_duplicate(
                file_hash=file_hash,
                content_hash=content_hash,
                similarity_threshold=video_hash_config.HASH_SIMILARITY_THRESHOLD
            )
            
            if is_dup:
                similarity = matched_record.get("similarity_score", 100.0) if matched_record else 100.0
                logger.info(
                    f"Duplicate video detected via {match_type}",
                    extra={
                        "extra_fields": {
                            "request_id": request_id,
                            "match_type": match_type,
                            "similarity": similarity
                        }
                    }
                )
                return {
                    "passed": False,
                    "fail_reasons": [
                        f"Video already uploaded (detected via {match_type}, "
                        f"similarity: {similarity:.1f}%)"
                    ],
                    "request_id": request_id
                }
            
            # If background processing requested
            if background and settings.ENABLE_BACKGROUND_TASKS:
                task_id = await submit_video_analysis_task(
                    video_path=str(video_path),
                    model_id=model_id
                )
                
                logger.info(f"Background task submitted", extra={
                    "extra_fields": {"request_id": request_id, "task_id": task_id}
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
                model_id=model_id,
                product_name=product_name,
                person_repo=person_repo,
                upload_repo=upload_repo,
                request_id=request_id,
                video_hashes=video_hashes
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
                "model_id": model_id,
                "success": result.get("passed", False)
            })
            
            result["request_id"] = request_id
            return result
            
        finally:
            # Clean up temporary file
            try:
                video_path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}", extra={
                    "extra_fields": {"request_id": request_id, "file_path": str(video_path)}
                })
    
    except HTTPException:
        raise
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        log_error(logger, e, request_id, {
            "model_id": model_id,
            "processing_time_ms": processing_time_ms
        })
        
        record_video_processing_metrics(
            duration_ms=processing_time_ms,
            file_size_mb=file.size / (1024 * 1024) if file.size else 0,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Video analysis failed",
                "message": "An unexpected error occurred during video processing",
                "request_id": request_id
            }
        )

async def _process_video_sync(
    video_path: Path,
    model_id: str,
    product_name: str,
    person_repo: PersonRepository,
    upload_repo: UploadRepository,
    request_id: str,
    video_hashes: Dict[str, Any]
) -> Dict[str, Any]:
    """Process video synchronously"""
    
    # Create temp directory for processing
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Initialize services
        object_detector = ObjectDetector()
        transcription_service = TranscriptionService()
        face_service = FaceRecognitionService()
        
        # Extract metadata, audio, frames
        metadata_start = time.time()
        metadata = get_metadata(video_path)
        metadata_time = (time.time() - metadata_start) * 1000
        
        audio_start = time.time()
        audio_path = extract_audio(video_path, tmp_path / "audio.wav")
        audio_time = (time.time() - audio_start) * 1000
        
        frames_start = time.time()
        frames_dir = extract_frames(video_path, tmp_path / "frames")
        frames_time = (time.time() - frames_start) * 1000
        
        log_performance(logger, "metadata_extraction", metadata_time, request_id)
        log_performance(logger, "audio_extraction", audio_time, request_id)
        log_performance(logger, "frame_extraction", frames_time, request_id)
        
        # Check person presence
        face_start = time.time()
        face_emb = face_service.get_face_embedding(frames_dir)
        face_time = (time.time() - face_start) * 1000
        
        record_model_usage_metrics("face_recognition", face_time, face_emb is not None)
        log_performance(logger, "face_recognition", face_time, request_id)
        
        person_id = person_repo.find_by_embedding(face_emb)
        if not person_id:
            person_id = person_repo.create(face_emb)
        
        if face_emb is None or not face_emb.any():
            return {
                "passed": False,
                "fail_reasons": ["No person detected"],
                "person_id": person_id
            }
        
        # Detect objects
        detection_start = time.time()
        detection = object_detector.detect_objects(frames_dir, product_name)
        detection_time = (time.time() - detection_start) * 1000

        record_model_usage_metrics("object_detection", detection_time, detection["product_found"])
        log_performance(logger, "object_detection", detection_time, request_id)

        # 🔥 Reject when YOLO doesn't detect a person
        if not detection["person_found"]:
            return {
                "passed": False,
                "fail_reasons": ["No person detected in video"],
                "person_id": person_id
            }

        # Reject when product is not found
        if not detection["product_found"]:
            expected = ", ".join(detection["expected_labels"]) or "unknown object"
            return {
                "passed": False,
                "fail_reasons": [f"Expected product '{product_name}' ({expected}) not detected in video"],
                "person_id": person_id
            }
        
        # Check duplicate product uploads for this person
        duplicate_products = []
        for product in detection["products"]:
            product_key = f"{product}|{model_id}"
            if upload_repo.find_by_person_and_product(person_id, product_key):
                duplicate_products.append(product_key)
        
        if duplicate_products:
            return {
                "passed": False,
                "fail_reasons": [f"Already uploaded {p}" for p in duplicate_products],
                "person_id": person_id
            }
        
        # Check video duration
        if metadata["duration_sec"] < settings.MIN_DURATION_SEC:
            return {
                "passed": False,
                "fail_reasons": [f"Duration < {settings.MIN_DURATION_SEC}s"],
                "person_id": person_id
            }
        
        # Transcribe audio and check bad words
        transcript_start = time.time()
        transcript = transcription_service.transcribe(audio_path)
        transcript_time = (time.time() - transcript_start) * 1000

        record_model_usage_metrics("transcription", transcript_time, True)
        log_performance(logger, "transcription", transcript_time, request_id)

        # --- Handle multiple transcription output formats safely ---
        if isinstance(transcript.get("label"), str) and transcript["label"].lower() == "toxic":
            # Toxicity classification-based output
            bad_words = [transcript.get("text", "")]
        else:
            # Keyword-based or standard transcription output
            bad_words = transcript.get("abusive_words", []) + transcript.get("violent_words", [])

        if bad_words:
            return {
                "passed": False,
                "fail_reasons": [f"Bad words found: {bad_words}"],
                "person_id": person_id
            }

        
        # All checks passed, calculate score
        
        # Handle missing word_count safely
        word_count = transcript.get("word_count")
        if word_count is None and "text" in transcript:
            # Fallback: estimate by splitting the text
            word_count = len(transcript["text"].split())
        else:
            word_count = word_count or 0
        
        # Enhanced quality-based scoring or legacy scoring
        if settings.SCORE_ENABLE_QUALITY_ANALYSIS:
            # Get detection confidences (using simple heuristic based on detection results)
            face_confidence = 0.8 if (face_emb is not None and face_emb.any()) else 0.0
            product_confidence = 0.8 if detection["product_found"] else 0.0
            
            # Initialize quality service and analyze
            quality_service = VideoQualityService()
            quality_metrics = quality_service.analyze_quality(
                video_path=video_path,
                frames_dir=frames_dir,
                audio_path=audio_path,
                metadata=metadata,
                face_confidence=face_confidence,
                product_confidence=product_confidence
            )
            
            # Base score
            score = settings.SCORE_BASE
            
            # Quality metrics (weighted)
            score += quality_metrics["technical_score"] * settings.SCORE_TECHNICAL_WEIGHT
            score += quality_metrics["visual_score"] * settings.SCORE_VISUAL_WEIGHT
            score += quality_metrics["content_score"] * settings.SCORE_CONTENT_WEIGHT
            score += quality_metrics["audio_score"] * settings.SCORE_AUDIO_WEIGHT
            
            # Legacy bonuses (optional)
            duration_bonus = max(metadata.get("duration_sec", 0) - settings.MIN_DURATION_SEC, 0) * settings.SCORE_DURATION_WEIGHT
            word_count_bonus = max(word_count - settings.MIN_WORDS, 0) * settings.SCORE_WORD_COUNT_WEIGHT
            
            score += duration_bonus
            score += word_count_bonus
            score = round(score, 2)
        else:
            # Legacy scoring (backward compatible)
            quality_metrics = None
            score = 100.0
            score += max(metadata.get("duration_sec", 0) - settings.MIN_DURATION_SEC, 0) * 1.0
            score += max(word_count - settings.MIN_WORDS, 0) * 0.5
            score = round(score, 2)

        
        # Save uploads
        for product in detection["products"]:
            upload_repo.create({
                "person_id": person_id,
                "product_label": product,
                "product_key": f"{product}|{model_id}",
                "model_id": model_id,
                "video_hash": compute_hash(video_path),  # Legacy v1 field
                **video_hashes,  # Add v2 hash fields
                "score": score
            })
        
        # Build response with optional quality breakdown
        response = {
            "passed": True,
            "score": score,
            "max_score": 100,
            "fail_reasons": [],
            "person_id": person_id,
            "metadata": metadata,
            "detected_products": detection["products"],
            "transcript": transcript["text"]
        }
        
        # Add score breakdown if quality analysis is enabled
        if settings.SCORE_ENABLE_QUALITY_ANALYSIS and quality_metrics:
            response["score_breakdown"] = {
                "technical": quality_metrics["technical_score"],
                "visual": quality_metrics["visual_score"],
                "content": quality_metrics["content_score"],
                "audio": quality_metrics["audio_score"],
                "duration_bonus": round(duration_bonus, 2),
                "word_count_bonus": round(word_count_bonus, 2)
            }
        
        return response

@router.get("/task/{task_id}")
async def get_task_status_endpoint(
    task_id: str,
    request: Request,
    _: bool = Depends(verify_service_access)
):
    """Get background task status"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        task_status = await get_task_status(task_id)
        
        if not task_status:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Task not found",
                    "task_id": task_id,
                    "request_id": request_id
                }
            )
        
        task_status["request_id"] = request_id
        return task_status
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(logger, e, request_id, {"task_id": task_id})
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to get task status",
                "task_id": task_id,
                "request_id": request_id
            }
        )

@router.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "video_analysis"}