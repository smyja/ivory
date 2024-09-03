import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.openapi.utils import get_openapi
import logging
from dotenv import load_dotenv
from routes.api import router as api_router
from routes.datasets import router as dataset_router

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for TOGETHER_API_KEY
if not os.environ.get('TOGETHER_API_KEY'):
    logger.error("TOGETHER_API_KEY is not set in the environment variables")
    raise EnvironmentError("TOGETHER_API_KEY is not set. Please set this environment variable before running the application.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI app is starting up")
    logger.info("Registered routes during startup:")
    for route in app.routes:
        logger.info(f"Path: {route.path}, Methods: {route.methods}")
    yield
    logger.info("FastAPI app is shutting down")

app = FastAPI(lifespan=lifespan)

app.include_router(api_router)
app.include_router(dataset_router, prefix="/datasets", tags=["datasets"])

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Question Clustering and Dataset API",
        version="1.0.0",
        description="API for clustering questions, generating titles, and managing datasets",
        routes=app.routes,
    )
    
    logger.info("Custom OpenAPI schema generation:")
    for route in app.routes:
        logger.info(f"Path: {route.path}, Methods: {route.methods}, Name: {route.name}")
        
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)