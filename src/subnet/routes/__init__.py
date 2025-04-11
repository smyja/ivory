from fastapi import APIRouter

from .dataset_routes import router as datasets_router

api_router = APIRouter()
api_router.include_router(datasets_router, prefix="/datasets", tags=["datasets"])
