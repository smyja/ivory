from fastapi import APIRouter

from .dataset_routes import router as datasets_router
from .query import router as query_router
from .meta import router as meta_router
from .api import router as legacy_core_router

api_router = APIRouter()
api_router.include_router(datasets_router, prefix="/datasets", tags=["datasets"])
api_router.include_router(query_router)
api_router.include_router(meta_router)
api_router.include_router(legacy_core_router)
