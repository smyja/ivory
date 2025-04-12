from fastapi import APIRouter

from .download import router as download_router
from .clustering import router as clustering_router
from .huggingface import router as huggingface_router
from .listing import router as listing_router
from .status import router as status_router

router = APIRouter()

# Include all the sub-routers with updated prefixes
router.include_router(listing_router)  # Base routes for listing go at the root
router.include_router(download_router, prefix="/download")
router.include_router(clustering_router, prefix="/clustering")
router.include_router(huggingface_router, prefix="/huggingface")
router.include_router(status_router)  # Status routes at the root level for /datasets/{id}/status
