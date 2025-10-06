from fastapi import APIRouter

router = APIRouter(prefix="/meta", tags=["meta"])


@router.get("/version")
def get_version():
    return {"api": "v1"}

