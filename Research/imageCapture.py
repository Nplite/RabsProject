from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from api.core.oauth2 import get_current_user
from api.core.db import db
import base64
import cv2
import time
import os

# Create API Router
router = APIRouter(tags=["RTSP Count Services"])

# Define Pydantic model for incoming RTSP link request
class RTSPLinkRequest(BaseModel):
    rtsp_link: str

# Global dictionary to track active streams
active_streams = {}

async def get_user_rtsp_links(current_user: dict):
    """
    Retrieve RTSP links for the authenticated user from the database.
    """
    user = await db.users.find_one({"_id": current_user["_id"]})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user.get("rtsp_links", [])

@router.post("/add_rtsp_link")
async def add_rtsp_link(request: RTSPLinkRequest, current_user: dict = Depends(get_current_user)):
    """
    Add an RTSP link to the user's database record.
    """
    try:
        rtsp_links = await get_user_rtsp_links(current_user)
        if request.rtsp_link in rtsp_links:
            return {"message": "RTSP link already exists"}
        rtsp_links.append(request.rtsp_link)
        await db.users.update_one({"_id": current_user["_id"]}, {"$set": {"rtsp_links": rtsp_links}})
        return {"message": "RTSP link added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add RTSP link: {str(e)}")

@router.get("/access_rtsp_stream")
async def access_rtsp_stream(current_user: dict = Depends(get_current_user)):
    """
    Retrieve the RTSP link for the authenticated user and store it in a temporary file.
    """
    try:
        rtsp_links = await get_user_rtsp_links(current_user)
        if not rtsp_links:
            raise HTTPException(status_code=404, detail="No RTSP links found for this user")
        
        # Ensure the /tmp directory exists
        os.makedirs("/tmp", exist_ok=True)
        
        # Write the first RTSP link to a temp file
        temp_file_path = f"/tmp/{current_user['email'].replace('@', '_')}_rtsp.txt"
        with open(temp_file_path, "w") as f:
            f.write(rtsp_links[0])
        
        return {"message": "RTSP link stored successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to access RTSP link: {str(e)}")

@router.get("/stream_video")
async def stream_video():
    """
    Stream live video using the RTSP link stored in the temporary file.
    """
    try:
        temp_files = [f for f in os.listdir("/tmp") if f.endswith("_rtsp.txt")]
        if not temp_files:
            raise HTTPException(status_code=404, detail="No RTSP link found in temp storage")
        
        temp_file_path = f"/tmp/{temp_files[0]}"
        with open(temp_file_path, "r") as f:
            rtsp_link = f.read().strip()
        
        print(f"Streaming from RTSP link: {rtsp_link}")
        cap = cv2.VideoCapture(rtsp_link)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail=f"Unable to open RTSP stream: {rtsp_link}")
        
        def generate_frames():
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    _, buffer = cv2.imencode('.jpg', cv2.resize(frame, (640, 480)))
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            finally:
                cap.release()
        
        return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stream video: {str(e)}")

@router.get("/capture_frame")
async def capture_frame():
    """
    Capture a frame from the stored RTSP link and return it as Base64.
    """
    try:
        temp_files = [f for f in os.listdir("/tmp") if f.endswith("_rtsp.txt")]
        if not temp_files:
            raise HTTPException(status_code=404, detail="No RTSP link found in temp storage")
        
        temp_file_path = f"/tmp/{temp_files[0]}"
        with open(temp_file_path, "r") as f:
            rtsp_link = f.read().strip()
        
        print(f"Capturing frame from RTSP link: {rtsp_link}")
        cap = cv2.VideoCapture(rtsp_link)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail=f"Unable to open RTSP stream: {rtsp_link}")
        
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise HTTPException(status_code=500, detail="Failed to capture frame")
        
        _, buffer = cv2.imencode('.jpg',frame)
        return base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to capture frame: {str(e)}")

@router.get("/stop_stream")
async def stop_stream():
    """
    Stop streaming the active RTSP link.
    """
    try:
        temp_files = [f for f in os.listdir("/tmp") if f.endswith("_rtsp.txt")]
        if not temp_files:
            return {"message": "No active stream found"}
        
        temp_file_path = temp_files[0]
        if temp_file_path in active_streams:
            active_streams[temp_file_path].release()
            active_streams.pop(temp_file_path, None)
            return {"message": "Stream stopped successfully"}
        
        return {"message": "No active stream found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop stream: {str(e)}")


