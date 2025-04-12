import asyncio
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Simple in-memory manager for download progress queues
# Note: This won't work correctly with multiple Uvicorn workers.
# For multi-worker setups, an external store like Redis Pub/Sub would be needed.
progress_queues: Dict[int, asyncio.Queue] = {}


async def register_dataset(dataset_id: int):
    """Register a dataset ID and create a queue for its progress."""
    if dataset_id not in progress_queues:
        progress_queues[dataset_id] = asyncio.Queue()
        logger.info(f"Registered progress queue for dataset_id: {dataset_id}")
    else:
        # Clear existing queue if re-registering (e.g., retry)
        # This might happen if a download is restarted
        while not progress_queues[dataset_id].empty():
            try:
                progress_queues[dataset_id].get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.info(f"Cleared existing progress queue for dataset_id: {dataset_id}")


async def report_progress(
    dataset_id: int,
    status: str,
    message: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
):
    """Report progress update for a given dataset ID."""
    if dataset_id in progress_queues:
        update = {"status": status}
        if message:
            update["message"] = message
        if data:
            update["data"] = data  # Include extra data like final status
        try:
            # Use put_nowait for potentially non-async contexts, handle QueueFull if needed
            # In background tasks, direct await should be fine
            await progress_queues[dataset_id].put(update)
            logger.debug(f"Reported progress for {dataset_id}: {update}")
        except Exception as e:
            # Log error but don't crash the download process
            logger.error(f"Failed to put progress update for {dataset_id}: {e}")


async def get_progress_queue(dataset_id: int) -> Optional[asyncio.Queue]:
    """Get the progress queue for a dataset ID."""
    return progress_queues.get(dataset_id)


async def unregister_dataset(dataset_id: int):
    """Remove the progress queue for a dataset ID upon completion or failure."""
    if dataset_id in progress_queues:
        # Optional: Ensure queue is empty before deleting?
        # Add a final "closed" message to signal stream end?
        # await report_progress(dataset_id, status="closed", message="Stream finished.")
        del progress_queues[dataset_id]
        logger.info(f"Unregistered progress queue for dataset_id: {dataset_id}")


async def cleanup_queues():
    """Clean up all queues (e.g., on server shutdown)."""
    logger.info(f"Cleaning up {len(progress_queues)} progress queues...")
    progress_queues.clear()
