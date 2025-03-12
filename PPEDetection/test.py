
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from threading import Thread
import os, sys
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
from bcrypt import checkpw


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from RabsProject.mongodb import MongoDBHandlerSaving
from RabsProject.camera_system import MultiCameraSystem, SingleCameraSystem
from dotenv import load_dotenv
load_dotenv()

# Access the SECRET_KEY
secret_key = os.getenv("SECRET_KEY")
print(f"Secret Key: {secret_key}")
SECRET_KEY = secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 3
running_single_camera_systems = {} 

app = FastAPI()
mongo_handler = MongoDBHandlerSaving()
running_camera_systems = {}

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Password and JWT utilities
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Pydantic Models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

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

class ModelInput(BaseModel):
    model:str 

class User(BaseModel):
    name: str
    email: str
    password: str
    phone_no: str
    role: str
    cameras: list
    disabled: Optional[bool] = None



def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(email: str):
    user_data = mongo_handler.user_collection.find_one({"email": email}, {"_id": 0})
    if user_data:
        return User(**user_data)
    return None


def authenticate_user(email: str, password: str):
    user_data = mongo_handler.user_collection.find_one({"email": email})
    
    if not user_data:
        return False

    if not verify_password(password, user_data["password"]):
        return False

    return User(**user_data)  # Convert to Pydantic Model


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire, "sub": data["sub"]})  # Ensure "sub" is always included
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = get_user(email)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled is None:
        current_user.disabled = False  # Default to False if missing

    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# API Endpoints
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/add_new_user")
def add_user(user: UserInput):
    """API to add a new user to MongoDB"""
    existing_user = mongo_handler.user_collection.find_one({"email": user.email})

    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")


    user_data = {
        "name": user.name,
        "email": user.email,
        "password": user.password,  # Store hashed password
        "phone_no": user.phone_no,
        "role": user.role,
        "cameras": user.cameras,
        "disabled": False
    }

    success = mongo_handler.save_user_to_mongodb(user_data)

    if success:
        return {"message": "User added successfully", "email": user.email}
    else:
        raise HTTPException(status_code=500, detail="Failed to add user")


@app.post("/add_camera")
def add_camera(data: CameraInput, current_user: User = Depends(get_current_active_user)):
    """Add a new camera dynamically"""
    # Verify user is adding camera to their own account or is an admin
    if current_user.email != data.email and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to add camera for this user")
        
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
def get_cameras(email: str, current_user: User = Depends(get_current_active_user)):
    """Get all cameras for a user"""
    # Verify user is requesting their own cameras or is an admin
    if current_user.email != email and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to view cameras for this user")
        
    cameras = mongo_handler.fetch_camera_rtsp_by_email(email)
    if cameras:
        return {"email": email, "cameras": cameras}
    else:
        raise HTTPException(status_code=404, detail="No cameras found for this user")
    

@app.delete("/remove_camera")
def remove_camera(data: CameraInput, current_user: User = Depends(get_current_active_user)):
    """Remove a camera dynamically"""
    if current_user.role != "admin" and current_user.email != data.email:
        raise HTTPException(status_code=403, detail="Not authorized to remove camera")

    existing_cameras = mongo_handler.fetch_camera_rtsp_by_email(data.email) or []
    updated_cameras = [cam for cam in existing_cameras if cam["camera_id"] != data.camera_id]

    if len(updated_cameras) == len(existing_cameras):
        raise HTTPException(status_code=404, detail="Camera not found")

    update_status = mongo_handler.save_user_to_mongodb({"email": data.email, "cameras": updated_cameras})

    if update_status:
        logger.info(f"Camera {data.camera_id} removed successfully for {data.email}")
        return {"message": "Camera removed successfully", "email": data.email, "camera_id": data.camera_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to remove camera")



@app.post("/start_streaming")
async def start_streaming( model_input: ModelInput,current_user: User = Depends(get_current_active_user)):
    """Start multi-camera streaming in a grid and return the streaming URL with embedded token"""
    global running_camera_systems

    if current_user.email in running_camera_systems:
        raise HTTPException(status_code=400, detail="Streaming is already running for this user")

    # Start the streaming system
    camera_system = MultiCameraSystem(email=current_user.email, model_path=model_input.model)
    running_camera_systems[current_user.email] = camera_system
    
    logger.info(f"Streaming started for {current_user.email}")

    # Create a special long-lived token for streaming
    streaming_token_expires = timedelta(hours=24)  # Adjust duration as needed
    streaming_token = create_access_token(
        data={"sub": current_user.email, "purpose": "streaming"},
        expires_delta=streaming_token_expires
    )

    # Generate the streaming URL with the token
    server_address = "http://0.0.0.0:8010"  # Update with your actual server address
    stream_url = f"{server_address}/stream?token={streaming_token}"
    
    return {
        "message": "Streaming started",
        "stream_url": stream_url,
        "token": streaming_token  # Include token in response for client-side use
    }



@app.get("/stream")
async def stream_video(
    token: str = None,
    authorization: str = None
):
    """Enhanced video streaming endpoint with flexible token handling"""
    try:
        # Try to get token from query parameter first
        streaming_token = token
        
        # If no query token, try to get from Authorization header
        if not streaming_token and authorization:
            if authorization.startswith("Bearer "):
                streaming_token = authorization.split(" ")[1]
        
        if not streaming_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No valid token provided",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Verify token and get user
        try:
            payload = jwt.decode(streaming_token, SECRET_KEY, algorithms=[ALGORITHM])
            token_email = payload.get("sub")
            if not token_email:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token format"
                )
            
            # Get user from token
            user = get_user(token_email)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found" )

        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Check if streaming is active
        if user.email not in running_camera_systems:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Streaming not found for this user"
            )
        
        camera_system = running_camera_systems[user.email]
        return StreamingResponse(
            camera_system.get_video_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )

    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Streaming error occurred"
        )
  


@app.get("/running_cameras")
async def running_cameras(current_user: User = Depends(get_current_user)):
    """Check which cameras are currently streaming"""
    global running_camera_systems
    active_streams = {
        email: {
            "cameras": list(system.camera_processors.keys())
        } for email, system in running_camera_systems.items()
    }
    return {"active_streams": active_streams}
  

@app.post("/stop_streaming")
async def stop_streaming(current_user: User = Depends(get_current_user)):
    global running_camera_systems
    if current_user.email not in running_camera_systems:
        raise HTTPException(status_code=404, detail="No active streaming session found")
    running_camera_systems[current_user.email].stop()
    del running_camera_systems[current_user.email]
    logger.info(f"Streaming stopped for {current_user.email}")
    return {"message": "Streaming stopped successfully"}


@app.post("/start_single_camera_streaming")
async def start_single_camera_streaming(
    camera_id: str,
    rtsp_link: str,
    current_user: User = Depends(get_current_active_user)
):
    """Start single-camera streaming and return the streaming URL with an embedded token"""
    global running_single_camera_systems

    if current_user.email in running_single_camera_systems:
        raise HTTPException(status_code=400, detail="Streaming is already running for this user")

    # Start the single camera system
    camera_system = SingleCameraSystem(
        camera_id=camera_id,
        rtsp_link=rtsp_link,
        email=current_user.email,
        model_path="FireDetection/fire_detection.pt"
    )
    camera_system.start()
    running_single_camera_systems[current_user.email] = camera_system

    logging.info(f"Single-camera streaming started for {current_user.email}")

    # Generate a streaming token
    streaming_token_expires = timedelta(hours=24)  # Adjust duration as needed
    streaming_token = create_access_token(
        data={"sub": current_user.email, "purpose": "single_camera_streaming"},
        expires_delta=streaming_token_expires
    )

    # Generate the streaming URL with the token
    server_address = "http://0.0.0.0:8010"  # Update with your actual server address
    stream_url = f"{server_address}/single_camera_stream?token={streaming_token}"
    
    return {
        "message": "Single camera streaming started",
        "stream_url": stream_url,
        "token": streaming_token  # Include token in response for client-side use
    }


@app.get("/single_camera_stream")
async def single_camera_stream(
    token: str = None,
    authorization: str = None
):
    """Video streaming endpoint for a single camera with token-based authentication"""
    try:
        # Extract token from query parameter or Authorization header
        streaming_token = token if token else authorization.split(" ")[1] if authorization and authorization.startswith("Bearer ") else None
        
        if not streaming_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No valid token provided",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Verify token and extract user email
        try:
            payload = jwt.decode(streaming_token, SECRET_KEY, algorithms=[ALGORITHM])
            token_email = payload.get("sub")
            if not token_email:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token format"
                )
            
            user = get_user(token_email)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )

        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Check if streaming is active
        if user.email not in running_single_camera_systems:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Streaming not found for this user"
            )
        
        camera_system = running_single_camera_systems[user.email]
        return StreamingResponse(
            camera_system.get_video_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )

    except Exception as e:
        logging.error(f"Streaming error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Streaming error occurred"
        )


@app.post("/stop_single_camera_streaming")
async def stop_single_camera_streaming(
    current_user: User = Depends(get_current_active_user)
):
    """Stop the single-camera stream for the current user"""
    global running_single_camera_systems

    if current_user.email not in running_single_camera_systems:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active streaming session found"
        )

    camera_system = running_single_camera_systems.pop(current_user.email)
    camera_system.stop()
    logging.info(f"Single-camera streaming stopped for {current_user.email}")

    return {"message": "Single camera streaming stopped"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)






