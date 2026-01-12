"""
Enhanced dependencies with connection pooling and error handling
"""
import asyncio
from typing import Generator, Optional
from contextlib import asynccontextmanager
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from app.core.config import settings
from app.core.logging import get_logger, log_error

logger = get_logger(__name__)

# Global MongoDB client
_mongo_client: Optional[MongoClient] = None
_mongo_client_lock = asyncio.Lock()

class DatabaseError(Exception):
    """Custom database exception"""
    pass

async def get_mongo_client() -> MongoClient:
    """Get MongoDB client with connection pooling"""
    global _mongo_client
    
    if _mongo_client is None:
        async with _mongo_client_lock:
            if _mongo_client is None:
                try:
                    _mongo_client = MongoClient(
                        settings.MONGODB_URL,
                        maxPoolSize=settings.MONGODB_MAX_POOL_SIZE,
                        serverSelectionTimeoutMS=settings.MONGODB_TIMEOUT_MS,
                        connectTimeoutMS=settings.MONGODB_TIMEOUT_MS,
                        socketTimeoutMS=settings.MONGODB_TIMEOUT_MS,
                        retryWrites=True,
                        retryReads=True
                    )
                    
                    # Test connection
                    _mongo_client.admin.command('ping')
                    logger.info("MongoDB connection established successfully")
                    
                except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                    log_error(logger, e, context={"operation": "mongodb_connection"})
                    raise DatabaseError(f"Failed to connect to MongoDB: {str(e)}")
    
    return _mongo_client

async def get_database():
    """Get database instance"""
    try:
        client = await get_mongo_client()
        return client[settings.MONGODB_DB_NAME]
    except Exception as e:
        log_error(logger, e, context={"operation": "get_database"})
        raise DatabaseError(f"Failed to get database: {str(e)}")


@asynccontextmanager
async def get_database_session():
    """Context manager for database operations with error handling"""
    try:
        db = get_database()
        yield db
    except Exception as e:
        log_error(logger, e, context={"operation": "database_session"})
        raise DatabaseError(f"Database operation failed: {str(e)}")

async def health_check_database() -> dict:
    """Check database health"""
    try:
        client = await get_mongo_client()
        
        # Test basic operations
        db = client[settings.MONGODB_DB_NAME]
        
        # Ping test
        ping_result = client.admin.command('ping')
        
        # Test write/read
        test_collection = db['health_check']
        test_doc = {'test': 'health_check', 'timestamp': 'now'}
        
        # Insert test document
        insert_result = test_collection.insert_one(test_doc)
        
        # Read test document
        found_doc = test_collection.find_one({'_id': insert_result.inserted_id})
        
        # Clean up test document
        test_collection.delete_one({'_id': insert_result.inserted_id})
        
        return {
            "status": "healthy",
            "ping": ping_result.get('ok') == 1,
            "write_test": insert_result.acknowledged,
            "read_test": found_doc is not None,
            "database": settings.MONGODB_DB_NAME,
            "connection_pool_size": client.max_pool_size
        }
        
    except Exception as e:
        log_error(logger, e, context={"operation": "database_health_check"})
        return {
            "status": "unhealthy",
            "error": str(e),
            "database": settings.MONGODB_DB_NAME
        }

async def close_database_connections():
    """Close all database connections"""
    global _mongo_client
    
    if _mongo_client:
        try:
            _mongo_client.close()
            logger.info("MongoDB connections closed")
        except Exception as e:
            log_error(logger, e, context={"operation": "close_database_connections"})
        finally:
            _mongo_client = None