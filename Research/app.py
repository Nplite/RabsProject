from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import yaml
from typing import Dict, List, Optional
import json
from pathlib import Path
from Rabs.camera_system import MultiCameraSystem, CameraProcessor, CameraStream
from Rabs.exception import RabsException
from Rabs.logger import logging


app = FastAPI(title="Camera Management System")

class CameraConfig(BaseModel):
    Camera_id: int
    RTSP_URL: str
    name: Optional[str] = None

class CameraResponse(BaseModel):
    camera_id: int
    status: str
    RTSP_URL: str
    name: Optional[str] = None

# Global camera system instance
camera_system = None
config_path = "config.yaml"

def save_config(config: dict):
    """Save the current configuration to yaml file"""
    try:
        with open(config_path, 'w') as file:
            yaml.dump(config, file)
        logging.info("Configuration saved successfully")
    except RabsException as e:
        logging.error(f"Failed to save configuration: {str(e)}")
        raise

def load_config() -> dict:
    """Load configuration from yaml file"""
    try:
        if not Path(config_path).exists():
            default_config = {"cameras": {}}
            save_config(default_config)
            return default_config
            
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except RabsException as e:
        logging.error(f"Failed to load configuration: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize the camera system on startup"""
    global camera_system
    try:
        if not camera_system:
                camera_system = MultiCameraSystem(config_path)
                camera_system.start()
        logging.info("Camera system initialized successfully")
    except RabsException as e:
        logging.error(f"Failed to initialize camera system: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global camera_system
    if camera_system:
        camera_system.stop()
        logging.info("Camera system shut down successfully")


@app.post("/cameras/", response_model=CameraResponse)
async def add_camera(camera: CameraConfig, background_tasks: BackgroundTasks):
    """Add a new camera to the system"""
    try:
        config = load_config()
        config['cameras'][f'camera_{camera.Camera_id}'] = {
            'RTSP_URL': camera.RTSP_URL,
            'name': camera.name
        }
        save_config(config)

        processor = CameraProcessor(camera.Camera_id, config['cameras'][f'camera_{camera.Camera_id}'])
        processor.stream.start()
        camera_system.camera_processors[camera.Camera_id] = processor
        
        return CameraResponse(
            camera_id=camera.Camera_id,
            status="active",
            RTSP_URL=camera.RTSP_URL,
            name=camera.name )
        
    except RabsException as e:
        logging.error(f"Failed to add camera: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cameras/{camera_id}")
async def remove_camera(camera_id: int):
    """Remove a camera from the system"""
    try:
        # Check if camera exists
        if camera_id not in camera_system.camera_processors:
            raise HTTPException(status_code=404, detail="Camera not found")
            
        # Stop the camera
        processor = camera_system.camera_processors[camera_id]
        processor.stream.stop()
        
        # Remove from camera processors
        del camera_system.camera_processors[camera_id]
        
        # Update configuration
        config = load_config()
        del config['cameras'][f'camera_{camera_id}']
        save_config(config)
        
        return {"message": f"Camera {camera_id} removed successfully"}
        
    except HTTPException:
        raise
    except RabsException as e:
        logging.error(f"Failed to remove camera: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cameras/", response_model=List[CameraResponse])
async def list_cameras():
    """List all active cameras"""
    try:
        cameras = []
        for camera_id, processor in camera_system.camera_processors.items():
            camera_config = processor.config
            cameras.append(
                CameraResponse(
                    camera_id=camera_id,
                    status="active" if not processor.stream.stopped else "inactive",
                    RTSP_URL=camera_config['RTSP_URL'],
                    name=camera_config.get('name')
                )
            )
        return cameras
    except RabsException as e:
        logging.error(f"Failed to list cameras: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cameras/{camera_id}", response_model=CameraResponse)
async def get_camera(camera_id: int):
    """Get details of a specific camera"""
    try:
        if camera_id not in camera_system.camera_processors:
            raise HTTPException(status_code=404, detail="Camera not found")
            
        processor = camera_system.camera_processors[camera_id]
        return CameraResponse(
            camera_id=camera_id,
            status="active" if not processor.stream.stopped else "inactive",
            RTSP_URL=processor.config['RTSP_URL'],
            name=processor.config.get('name')
        )
    except HTTPException:
        raise
    except RabsException as e:
        logging.error(f"Failed to get camera details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cameras/{camera_id}/restart")
async def restart_camera(camera_id: int):
    """Restart a specific camera"""
    try:
        if camera_id not in camera_system.camera_processors:
            raise HTTPException(status_code=404, detail="Camera not found")
            
        processor = camera_system.camera_processors[camera_id]
        processor.stream.stop()
        processor.stream = CameraStream(processor.config['RTSP_URL'], camera_id)
        processor.stream.start()
        
        return {"message": f"Camera {camera_id} restarted successfully"}
    except HTTPException:
        raise
    except RabsException as e:
        logging.error(f"Failed to restart camera: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
