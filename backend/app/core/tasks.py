"""
Background task processing for video analysis
"""
import asyncio
import uuid
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
from app.core.config import settings
from app.core.logging import get_logger, log_performance, log_error

logger = get_logger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

class BackgroundTask:
    """Background task representation"""
    
    def __init__(self, task_id: str, task_type: str, data: Dict[str, Any]):
        self.task_id = task_id
        self.task_type = task_type
        self.data = data
        self.status = TaskStatus.PENDING
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.progress: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "progress": self.progress
        }

class TaskManager:
    """Manages background tasks"""
    
    def __init__(self):
        self.tasks: Dict[str, BackgroundTask] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the task manager"""
        if settings.ENABLE_BACKGROUND_TASKS:
            self._cleanup_task = asyncio.create_task(self._cleanup_old_tasks())
            logger.info("Background task manager started")
    
    async def stop(self):
        """Stop the task manager"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all running tasks
        for task_id, task in self.running_tasks.items():
            task.cancel()
            logger.warning(f"Cancelled running task: {task_id}")
        
        logger.info("Background task manager stopped")
    
    async def submit_task(self, task_type: str, data: Dict[str, Any]) -> str:
        """Submit a new background task"""
        task_id = str(uuid.uuid4())
        task = BackgroundTask(task_id, task_type, data)
        
        self.tasks[task_id] = task
        
        if settings.ENABLE_BACKGROUND_TASKS:
            # Start task execution
            asyncio_task = asyncio.create_task(self._execute_task(task))
            self.running_tasks[task_id] = asyncio_task
        else:
            # Synchronous execution
            await self._execute_task(task)
        
        logger.info(f"Task submitted: {task_id} ({task_type})")
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        task = self.tasks.get(task_id)
        if task:
            return task.to_dict()
        return None
    
    async def _execute_task(self, task: BackgroundTask):
        """Execute a background task"""
        try:
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.utcnow()
            
            logger.info(f"Starting task execution: {task.task_id}")
            
            # Execute based on task type
            if task.task_type == "video_analysis":
                result = await self._execute_video_analysis(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.progress = 100.0
            
            logger.info(f"Task completed successfully: {task.task_id}")
            
        except asyncio.CancelledError:
            task.status = TaskStatus.TIMEOUT
            task.error = "Task was cancelled"
            logger.warning(f"Task cancelled: {task.task_id}")
            raise
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.utcnow()
            log_error(logger, e, context={"task_id": task.task_id, "task_type": task.task_type})
            
        finally:
            # Remove from running tasks
            self.running_tasks.pop(task.task_id, None)
    
    async def _execute_video_analysis(self, task: BackgroundTask) -> Dict[str, Any]:
        """Execute video analysis task"""
        # Import here to avoid circular imports
        from app.services.video_processor import VideoAnalysisService
        
        analysis_service = VideoAnalysisService()
        
        # Update progress
        task.progress = 10.0
        
        # Extract data
        video_path = task.data.get("video_path")
        model_id = task.data.get("model_id")
        person_id = task.data.get("person_id")
        
        if not all([video_path, model_id]):
            raise ValueError("Missing required data for video analysis")
        
        # Perform analysis
        result = await analysis_service.analyze_video_async(
            video_path=video_path,
            model_id=model_id,
            person_id=person_id,
            progress_callback=lambda p: setattr(task, 'progress', p)
        )
        
        return result
    
    async def _cleanup_old_tasks(self):
        """Clean up old completed tasks"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                tasks_to_remove = []
                
                for task_id, task in self.tasks.items():
                    if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] and 
                        task.completed_at and 
                        task.completed_at < cutoff_time):
                        tasks_to_remove.append(task_id)
                
                for task_id in tasks_to_remove:
                    del self.tasks[task_id]
                
                if tasks_to_remove:
                    logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error(logger, e, context={"operation": "cleanup_old_tasks"})

# Global task manager instance
task_manager = TaskManager()

async def submit_video_analysis_task(video_path: str, model_id: str, person_id: str = None) -> str:
    """Submit video analysis task"""
    data = {
        "video_path": video_path,
        "model_id": model_id,
        "person_id": person_id
    }
    
    return await task_manager.submit_task("video_analysis", data)

async def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task status"""
    return await task_manager.get_task_status(task_id)

async def start_task_manager():
    """Start the task manager"""
    await task_manager.start()

async def stop_task_manager():
    """Stop the task manager"""
    await task_manager.stop()
