import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.openapi.utils import get_openapi
import logging
from dotenv import load_dotenv
from routes import api_router
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Require Together API key (we only support Together now)
if not os.environ.get("TOGETHER_API_KEY"):
    logger.error("TOGETHER_API_KEY is not set in the environment variables")
    raise HTTPException(
        status_code=500,
        detail="TOGETHER_API_KEY is not set. Please set this environment variable before running the application.",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI app is starting up")
    yield
    logger.info("FastAPI app is shutting down")


app = FastAPI(lifespan=lifespan)

# Configure CORS
origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers (versioned path is canonical and enforced)
app.include_router(api_router, prefix="/api/v1")

# Optional compatibility alias if explicitly enabled
if os.environ.get("IVORY_ENABLE_UNVERSIONED_API", "0").lower() in {"1", "true", "yes", "on"}:
    app.include_router(api_router)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Clustering and Dataset API",
        version="1.0.0",
        description="API for clustering and managing datasets",
        routes=app.routes,
    )

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
