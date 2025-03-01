import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from threading import Thread
import time
from Rabs.mongodb import MongoDBHandlerSaving
from Rabs.camera_system import MultiCameraSystem 

app = FastAPI()
mongo_handler = MongoDBHandlerSaving()
running_camera_systems = {}

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class CameraInput(BaseModel):
    email: str
    camera_id: str
    rtsp_link: str

class UserInput(BaseModel):
    name: str
    email: str
    password: str
    phone_no: str
    role: str = "user"
    cameras: list = []

class EmailInput(BaseModel):
    email: str




@app.post("/add_new_user")
def add_user(user: UserInput):
    """API to add a new user to MongoDB"""
    existing_user = mongo_handler.user_collection.find_one({"email": user.email})

    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    user_data = {
        "name": user.name,
        "email": user.email,
        "password": mongo_handler.hash_password(user.password), 
        "phone_no": user.phone_no,
        "role": user.role,
        "cameras": user.cameras }

    success = mongo_handler.save_user_to_mongodb(user_data)

    if success:
        return {"message": "User added successfully", "email": user.email}
    else:
        raise HTTPException(status_code=500, detail="Failed to add user")


@app.post("/add_camera")
def add_camera(data: CameraInput):
    """Add a new camera dynamically"""
    existing_cameras = mongo_handler.fetch_camera_rtsp_by_email(data.email) or []

    # Check if camera_id already exists
    if any(cam["camera_id"] == data.camera_id for cam in existing_cameras):
        raise HTTPException(status_code=400, detail="Camera ID already exists")

    new_camera = {"camera_id": data.camera_id, "rtsp_link": data.rtsp_link}
    existing_cameras.append(new_camera)

    update_status = mongo_handler.save_user_to_mongodb({"email": data.email, "cameras": existing_cameras})

    if update_status:
        logger.info(f"Camera {data.camera_id} added successfully for {data.email}")
        return {"message": "Camera added successfully", "email": data.email, "camera_id": data.camera_id, "rtsp_link": data.rtsp_link}
    else:
        raise HTTPException(status_code=500, detail="Failed to add camera")
    
@app.get("/get_cameras")
def get_cameras(email: str):
    """Get all cameras for a user"""
    cameras = mongo_handler.fetch_camera_rtsp_by_email(email)
    if cameras:
        return {"email": email, "cameras": cameras}
    else:
        raise HTTPException(status_code=404, detail="No cameras found for this user")
    
@app.delete("/remove_camera")
def remove_camera(data: CameraInput):
    """Remove a camera dynamically"""
    existing_cameras = mongo_handler.fetch_camera_rtsp_by_email(data.email) or []

    # Filter out the camera to be removed
    updated_cameras = [cam for cam in existing_cameras if cam["camera_id"] != data.camera_id]

    if len(updated_cameras) == len(existing_cameras):
        raise HTTPException(status_code=404, detail="Camera not found")

    update_status = mongo_handler.save_user_to_mongodb({"email": data.email, "cameras": updated_cameras})

    if update_status:
        logger.info(f"Camera {data.camera_id} removed successfully for {data.email}")
        return {"message": "Camera removed successfully", "email": data.email, "camera_id": data.camera_id, "rtsp_link": data.rtsp_link}
    else:
        raise HTTPException(status_code=500, detail="Failed to remove camera")


@app.post("/start_streaming")
def start_streaming(data: EmailInput):
    """Start multi-camera streaming in a grid"""
    global running_camera_systems

    if data.email in running_camera_systems:
        raise HTTPException(status_code=400, detail="Streaming is already running for this user")

    camera_system = MultiCameraSystem(email=data.email)
    stream_thread = Thread(target=camera_system.start, daemon=True)
    stream_thread.start()
    running_camera_systems[data.email] = camera_system
    logger.info(f"Streaming started for {data.email}")

    return {"message": "Streaming started", "email": data.email}

@app.post("/stop_streaming")
def stop_streaming(data: EmailInput):
    """Stop multi-camera streaming"""
    global running_camera_systems

    if data.email not in running_camera_systems:
        raise HTTPException(status_code=400, detail="No active streaming found for this user")

    camera_system = running_camera_systems.pop(data.email)
    camera_system.stop()
    logger.info(f"Streaming stopped for {data.email}")

    return {"message": "Streaming stopped", "email": data.email}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
