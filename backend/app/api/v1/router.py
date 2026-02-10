from fastapi import APIRouter
from app.api.v1.endpoints import analysis, search

api_router = APIRouter()
api_router.include_router(analysis.router, prefix="/video", tags=["Video"])
api_router.include_router(search.router, prefix="/search", tags=["Search"])