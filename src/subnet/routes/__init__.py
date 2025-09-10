from fastapi import APIRouter

from .dataset_routes import router as datasets_router
from .query import router as query_router

api_router = APIRouter()
api_router.include_router(datasets_router, prefix="/datasets", tags=["datasets"])
api_router.include_router(query_router)
