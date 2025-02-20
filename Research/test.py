# from fastapi import FastAPI, HTTPException, Depends
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel
# from Rabs.mongodb import MongoDBHandlerSaving
# import base64
# import cv2
# import os

# db = MongoDBHandlerSaving()
# app = FastAPI(tags=["RTSP Count Services"])

# # Fixing the get_current_user - it should be a function, not a string
# async def get_current_user():
#     return {"email": "pranavkumar82100@gmail.com"}

# class RTSPLinkRequest(BaseModel):
#     rtsp_link: str

# active_streams = {}

# async def get_authenticated_user(current_user: dict):
#     # Fixed to use the correct fetch_user_by_email method
#     user = db.fetch_user_by_email(current_user["email"])
#     if not user:
#         raise HTTPException(status_code=401, detail="Unauthorized: User not found")
#     return user

# @app.post("/add_rtsp_link")
# async def add_rtsp_link(request: RTSPLinkRequest, current_user: dict = Depends(get_current_user)):
#     try:
#         user = await get_authenticated_user(current_user)
#         rtsp_links = user.get("rtsp_links", [])

#         if request.rtsp_link in rtsp_links:
#             return {"message": "RTSP link already exists"}

#         rtsp_links.append(request.rtsp_link)
#         # Fixed to use the correct collection
#         db.user_collection.update_one(
#             {"email": current_user["email"]}, 
#             {"$set": {"rtsp_links": rtsp_links}}
#         )
        
#         return {"message": "RTSP link added successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to add RTSP link: {str(e)}")

# @app.get("/access_rtsp_stream")
# async def access_rtsp_stream(current_user: dict = Depends(get_current_user)):
#     try:
#         user = await get_authenticated_user(current_user)
#         rtsp_links = user.get("rtsp_links", [])

#         if not rtsp_links:
#             raise HTTPException(status_code=404, detail="No RTSP links found for this user")

#         os.makedirs("/tmp", exist_ok=True)

#         temp_file_path = f"/tmp/{current_user['email'].replace('@', '_')}_rtsp.txt"
#         with open(temp_file_path, "w") as f:
#             f.write(rtsp_links[0])

#         return {"message": "RTSP link stored successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to access RTSP link: {str(e)}")

# @app.get("/stream_video")
# async def stream_video(current_user: dict = Depends(get_current_user)):
#     try:
#         user = await get_authenticated_user(current_user)
#         user_email = current_user['email']
        
#         temp_file_path = f"/tmp/{user_email.replace('@', '_')}_rtsp.txt"
#         if not os.path.exists(temp_file_path):
#             raise HTTPException(status_code=404, detail="No RTSP link found in temp storage")

#         with open(temp_file_path, "r") as f:
#             rtsp_link = f.read().strip()

#         print(f"Streaming from RTSP link: {rtsp_link}")
#         cap = cv2.VideoCapture(rtsp_link)
#         if not cap.isOpened():
#             raise HTTPException(status_code=400, detail=f"Unable to open RTSP stream: {rtsp_link}")
            
#         # Store the cap in active_streams for later cleanup
#         active_streams[temp_file_path] = cap

#         def generate_frames():
#             try:
#                 while cap.isOpened():
#                     ret, frame = cap.read()
#                     if not ret:
#                         break
#                     _, buffer = cv2.imencode('.jpg', cv2.resize(frame, (640, 480)))
#                     yield (b"--frame\r\n"
#                            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
#             finally:
#                 cap.release()
#                 active_streams.pop(temp_file_path, None)

#         return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to stream video: {str(e)}")

# @app.get("/capture_frame")
# async def capture_frame(current_user: dict = Depends(get_current_user)):
#     try:
#         user = await get_authenticated_user(current_user)

#         temp_file_path = f"/tmp/{current_user['email'].replace('@', '_')}_rtsp.txt"
#         if not os.path.exists(temp_file_path):
#             raise HTTPException(status_code=404, detail="No RTSP link found in temp storage")

#         with open(temp_file_path, "r") as f:
#             rtsp_link = f.read().strip()

#         print(f"Capturing frame from RTSP link: {rtsp_link}")
#         cap = cv2.VideoCapture(rtsp_link)
#         if not cap.isOpened():
#             raise HTTPException(status_code=400, detail=f"Unable to open RTSP stream: {rtsp_link}")

#         ret, frame = cap.read()
#         cap.release()
#         if not ret:
#             raise HTTPException(status_code=500, detail="Failed to capture frame")

#         _, buffer = cv2.imencode('.jpg', frame)
#         return {"frame": base64.b64encode(buffer).decode("utf-8")}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to capture frame: {str(e)}")

# @app.get("/stop_stream")
# async def stop_stream(current_user: dict = Depends(get_current_user)):
#     try:
#         temp_file_path = f"/tmp/{current_user['email'].replace('@', '_')}_rtsp.txt"
#         if not os.path.exists(temp_file_path):
#             return {"message": "No active stream found"}

#         if temp_file_path in active_streams:
#             active_streams[temp_file_path].release()
#             active_streams.pop(temp_file_path, None)
#             os.remove(temp_file_path)
#             return {"message": "Stream stopped successfully"}

#         return {"message": "No active stream found"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to stop stream: {str(e)}")
    

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


























from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import logging
from pymongo import MongoClient
from Rabs.camera_system import MultiCameraSystem, CameraProcessor, CameraStream
from Rabs.exception import RabsException
from Rabs.logger import logging

# MongoDB setup
MONGO_URL = "mongodb://localhost:27017"
DB_NAME = "RabsProject"
USER_COLLECTION = "UserAuth"

client = MongoClient(MONGO_URL)
db = client[DB_NAME]
user_collection = db[USER_COLLECTION]

app = FastAPI(title="Camera Management System")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

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

def get_user(email: str):
    return user_collection.find_one({"email": email}, {"_id": 0})

def authenticate_user(email: str, password: str):
    user = get_user(email)
    if user and user["password"] == password:
        return user
    return None

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    return {"access_token": user["email"], "token_type": "bearer"}

@app.on_event("startup")
async def startup_event():
    global camera_system
    try:
        if not camera_system:
            camera_system = MultiCameraSystem()
            camera_system.start()
        logging.info("Camera system initialized successfully")
    except RabsException as e:
        logging.error(f"Failed to initialize camera system: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    global camera_system
    if camera_system:
        camera_system.stop()
        logging.info("Camera system shut down successfully")

@app.post("/cameras/", response_model=CameraResponse)
async def add_camera(camera: CameraConfig, token: str = Depends(oauth2_scheme)):
    user = get_user(token)
    if not user:
        raise HTTPException(status_code=403, detail="Unauthorized access")
    if camera.RTSP_URL not in user.get("rtsp_links", []):
        raise HTTPException(status_code=403, detail="No access to this RTSP link")
    
    processor = CameraProcessor(camera.Camera_id, {"RTSP_URL": camera.RTSP_URL, "name": camera.name})
    processor.stream.start()
    camera_system.camera_processors[camera.Camera_id] = processor
    
    return CameraResponse(
        camera_id=camera.Camera_id,
        status="active",
        RTSP_URL=camera.RTSP_URL,
        name=camera.name )

@app.get("/cameras/", response_model=List[CameraResponse])
async def list_cameras(token: str = Depends(oauth2_scheme)):
    user = get_user(token)
    if not user:
        raise HTTPException(status_code=403, detail="Unauthorized access")
    
    cameras = []
    for camera_id, processor in camera_system.camera_processors.items():
        if processor.config["RTSP_URL"] in user.get("rtsp_links", []):
            cameras.append(CameraResponse(
                camera_id=camera_id,
                status="active" if not processor.stream.stopped else "inactive",
                RTSP_URL=processor.config['RTSP_URL'],
                name=processor.config.get('name')
            ))
    return cameras

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

