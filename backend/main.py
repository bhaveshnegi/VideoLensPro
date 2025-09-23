# main.py - FastAPI backend with video analysis pipeline
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import uuid
import asyncio
import json
from datetime import datetime
from pathlib import Path
import aiofiles
from typing import Dict, List, Optional
import subprocess

# Video processing imports (you'll need to install these)
# pip install opencv-python moviepy tensorflow numpy pillow

import cv2
import numpy as np
from moviepy.editor import VideoFileClip

app = FastAPI(title="Video Analysis Pipeline API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://video-lens-pro.vercel.app"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
TEMP_DIR = Path("temp")

# Create directories
for dir_path in [UPLOAD_DIR, RESULTS_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

# In-memory storage for job status (use Redis/DB in production)
job_status: Dict[str, Dict] = {}

class VideoPipeline:
    """Video analysis pipeline with multiple processing steps"""
    
    def __init__(self, video_path: str, job_id: str):
        self.video_path = video_path
        self.job_id = job_id
        self.results = {}
    
    async def update_status(self, step: str, progress: int, message: str = ""):
        """Update job status"""
        job_status[self.job_id].update({
            "current_step": step,
            "progress": progress,
            "message": message,
            "updated_at": datetime.now().isoformat()
        })
    
    async def step_1_basic_info(self):
        """Step 1: Extract basic video information"""
        await self.update_status("basic_info", 10, "Extracting basic video information...")
        
        cap = cv2.VideoCapture(self.video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        self.results["basic_info"] = {
            "fps": fps,
            "frame_count": frame_count,
            "resolution": {"width": width, "height": height},
            "duration_seconds": duration,
            "file_size_mb": os.path.getsize(self.video_path) / (1024 * 1024)
        }
        
        await self.update_status("basic_info", 20, "Basic info extracted")
    
    async def step_2_frame_analysis(self):
        """Step 2: Analyze frames for motion, brightness, etc."""
        await self.update_status("frame_analysis", 30, "Analyzing frames...")
        
        cap = cv2.VideoCapture(self.video_path)
        
        frame_data = []
        frame_count = 0
        total_brightness = 0
        motion_scores = []
        prev_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate brightness
            brightness = np.mean(gray)
            total_brightness += brightness
            
            # Calculate motion (if previous frame exists)
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                motion_score = np.mean(diff)
                motion_scores.append(motion_score)
            
            prev_frame = gray.copy()
            frame_count += 1
            
            # Sample every 30th frame to avoid too much data
            if frame_count % 30 == 0:
                await self.update_status("frame_analysis", 
                                       30 + (frame_count / self.results["basic_info"]["frame_count"]) * 30,
                                       f"Processed {frame_count} frames")
        
        cap.release()
        
        self.results["frame_analysis"] = {
            "average_brightness": total_brightness / frame_count if frame_count > 0 else 0,
            "motion_scores": {
                "average": np.mean(motion_scores) if motion_scores else 0,
                "max": np.max(motion_scores) if motion_scores else 0,
                "std": np.std(motion_scores) if motion_scores else 0
            },
            "frames_analyzed": frame_count
        }
        
        await self.update_status("frame_analysis", 60, "Frame analysis completed")
    
    async def step_3_scene_detection(self):
        """Step 3: Detect scene changes"""
        await self.update_status("scene_detection", 70, "Detecting scenes...")
        
        # Simple scene detection based on frame differences
        cap = cv2.VideoCapture(self.video_path)
        
        scenes = []
        prev_frame = None
        frame_num = 0
        scene_threshold = 30  # Adjust based on your needs
        current_scene_start = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                scene_change_score = np.mean(diff)
                
                if scene_change_score > scene_threshold:
                    # Scene change detected
                    scenes.append({
                        "start_frame": current_scene_start,
                        "end_frame": frame_num,
                        "duration": (frame_num - current_scene_start) / self.results["basic_info"]["fps"]
                    })
                    current_scene_start = frame_num
            
            prev_frame = gray
            frame_num += 1
        
        # Add the last scene
        if current_scene_start < frame_num:
            scenes.append({
                "start_frame": current_scene_start,
                "end_frame": frame_num,
                "duration": (frame_num - current_scene_start) / self.results["basic_info"]["fps"]
            })
        
        cap.release()
        
        self.results["scene_detection"] = {
            "total_scenes": len(scenes),
            "scenes": scenes[:10],  # Limit to first 10 scenes
            "average_scene_duration": np.mean([s["duration"] for s in scenes]) if scenes else 0
        }
        
        await self.update_status("scene_detection", 80, "Scene detection completed")
    
    async def step_4_generate_thumbnails(self):
        """Step 4: Generate thumbnails"""
        await self.update_status("thumbnails", 90, "Generating thumbnails...")
        
        cap = cv2.VideoCapture(self.video_path)
        total_frames = self.results["basic_info"]["frame_count"]
        
        # Generate 5 thumbnails at different time points
        thumbnail_times = [0.1, 0.25, 0.5, 0.75, 0.9]  # 10%, 25%, 50%, 75%, 90%
        thumbnails = []
        
        for i, time_ratio in enumerate(thumbnail_times):
            frame_number = int(total_frames * time_ratio)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if ret:
                # Resize thumbnail
                height, width = frame.shape[:2]
                aspect_ratio = width / height
                new_width = 320
                new_height = int(new_width / aspect_ratio)
                
                thumbnail = cv2.resize(frame, (new_width, new_height))
                
                # Save thumbnail
                thumbnail_filename = f"{self.job_id}_thumb_{i}.jpg"
                thumbnail_path = RESULTS_DIR / thumbnail_filename
                cv2.imwrite(str(thumbnail_path), thumbnail)
                
                thumbnails.append({
                    "filename": thumbnail_filename,
                    "timestamp": frame_number / self.results["basic_info"]["fps"],
                    "frame_number": frame_number
                })
        
        cap.release()
        
        self.results["thumbnails"] = thumbnails
        await self.update_status("thumbnails", 95, "Thumbnails generated")
    
    async def step_5_finalize(self):
        """Step 5: Finalize results"""
        await self.update_status("finalizing", 98, "Finalizing results...")
        
        # Save complete results to file
        results_filename = f"{self.job_id}_results.json"
        results_path = RESULTS_DIR / results_filename
        
        async with aiofiles.open(results_path, 'w') as f:
            await f.write(json.dumps(self.results, indent=2))
        
        self.results["results_file"] = results_filename
        
        await self.update_status("completed", 100, "Analysis completed successfully!")
        
        # Update final job status
        job_status[self.job_id].update({
            "status": "completed",
            "results": self.results,
            "completed_at": datetime.now().isoformat()
        })
    
    async def run_pipeline(self):
        """Run the complete pipeline"""
        try:
            await self.step_1_basic_info()
            await self.step_2_frame_analysis()
            await self.step_3_scene_detection()
            await self.step_4_generate_thumbnails()
            await self.step_5_finalize()
            
        except Exception as e:
            job_status[self.job_id].update({
                "status": "error",
                "error": str(e),
                "updated_at": datetime.now().isoformat()
            })

# API Endpoints

@app.post("/upload-video")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload video and start processing pipeline"""
    
    # Validate file
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'mp4'
    video_filename = f"{job_id}.{file_extension}"
    video_path = UPLOAD_DIR / video_filename
    
    async with aiofiles.open(video_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    # Initialize job status
    job_status[job_id] = {
        "job_id": job_id,
        "filename": file.filename,
        "status": "processing",
        "current_step": "initialized",
        "progress": 0,
        "created_at": datetime.now().isoformat(),
        "video_path": str(video_path)
    }
    
    # Start pipeline in background
    pipeline = VideoPipeline(str(video_path), job_id)
    background_tasks.add_task(pipeline.run_pipeline)
    
    return JSONResponse({
        "job_id": job_id,
        "message": "Video uploaded successfully. Processing started.",
        "status": "processing"
    })

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Get processing status of a job"""
    
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_status[job_id]

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Get detailed results of completed job"""
    
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_status[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    return job.get("results", {})

@app.get("/download-results/{job_id}")
async def download_results(job_id: str):
    """Download complete results as JSON file"""
    
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    results_filename = f"{job_id}_results.json"
    results_path = RESULTS_DIR / results_filename
    
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Results file not found")
    
    return FileResponse(
        path=results_path,
        filename=results_filename,
        media_type="application/json"
    )

@app.get("/thumbnail/{filename}")
async def get_thumbnail(filename: str):
    """Get thumbnail image"""
    
    thumbnail_path = RESULTS_DIR / filename
    
    if not thumbnail_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    
    return FileResponse(thumbnail_path, media_type="image/jpeg")

@app.get("/jobs")
async def list_jobs():
    """List all jobs"""
    return {"jobs": list(job_status.keys())}

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete job and associated files"""
    
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_status[job_id]
    
    # Clean up files
    if "video_path" in job:
        video_path = Path(job["video_path"])
        if video_path.exists():
            os.remove(video_path)
    
    # Remove results files
    results_file = RESULTS_DIR / f"{job_id}_results.json"
    if results_file.exists():
        os.remove(results_file)
    
    # Remove thumbnails
    for thumbnail_file in RESULTS_DIR.glob(f"{job_id}_thumb_*.jpg"):
        os.remove(thumbnail_file)
    
    # Remove from memory
    del job_status[job_id]
    
    return {"message": "Job deleted successfully"}

@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "Video Analysis Pipeline API",
        "version": "1.0.0",
        "status": "running"
    }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)