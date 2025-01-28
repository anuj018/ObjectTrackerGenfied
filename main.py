import asyncio
from fastapi import FastAPI
from datetime import datetime
from database import SessionLocal, ProcessedVideo
from azure_client import container_client
from processor import process_video
from sender import send_detection_data
import tempfile
import os
from concurrent.futures import ProcessPoolExecutor

app = FastAPI()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def monitor_blob_storage():
    # Create a ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        loop = asyncio.get_running_loop()
        while True:
            db = SessionLocal()
            try:
                blobs = container_client.list_blobs()
                for blob in blobs:
                    # Check if blob is already processed
                    processed = db.query(ProcessedVideo).filter(ProcessedVideo.blob_name == blob.name).first()
                    if not processed:
                        print(f"New video found: {blob.name}")
                        # Download the blob
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                            download_stream = container_client.download_blob(blob)
                            tmp_file.write(download_stream.readall())
                            temp_file_path = tmp_file.name

                        # Process the video in a separate process
                        detections = await loop.run_in_executor(executor, process_video, temp_file_path)

                        # Send the data
                        await send_detection_data(detections)

                        # Mark as processed
                        new_record = ProcessedVideo(
                            blob_name=blob.name,
                            processed_at=datetime.utcnow()
                        )
                        db.add(new_record)
                        db.commit()

                        # Remove the temporary file
                        os.remove(temp_file_path)
            except Exception as e:
                print(f"Error during processing: {e}")
            finally:
                db.close()
            # Wait for a specific interval before checking again
            await asyncio.sleep(30)  # Check every 30 seconds

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(monitor_blob_storage())

@app.get("/")
def read_root():
    return {"message": "Blob Monitor is running."}