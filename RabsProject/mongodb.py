
import logging
from pymongo import MongoClient
from bcrypt import hashpw, gensalt
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MongoDBHandlerSaving:
    def __init__(self, url="mongodb://localhost:27017", db_name="RabsProject", user_collection_name="UserAuth"):
        """Initialize the MongoDB client and select the database and collection."""
        try:
            self.client = MongoClient(url)
            self.db = self.client[db_name]
            self.user_collection = self.db[user_collection_name]
            logger.info("Connected to MongoDB successfully")
        except Exception as e:
            logger.exception("Error connecting to MongoDB")
            raise


    def close_connection(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


    @staticmethod
    def hash_password(password):
        """Hash the user's password before storing it."""
        return hashpw(password.encode(), gensalt()).decode()


    def save_user_to_mongodb(self, user_data):
        """Insert or update user data securely with camera details."""
        try:
            # Fetch existing user data
            existing_user = self.user_collection.find_one({"email": user_data["email"]})
            
            if existing_user:
                # If user exists, only update cameras without overwriting name, role, etc.
                filtered_user_data = {"cameras": user_data.get("cameras", [])}
            else:
                # If new user, ensure required fields exist
                filtered_user_data = {
                    "name": user_data.get("name", "Unknown"),  # Default to 'Unknown' if missing
                    "email": user_data["email"],

                    "password": self.hash_password(user_data.get("password", "defaultpassword")),  

                    "phone_no": user_data.get("phone_no", ""),
                    "role": user_data.get("role", "user"),
                    "cameras": user_data.get("cameras", [])  }

            result = self.user_collection.update_one(
                {"email": user_data["email"]},  # Match existing user by email
                {"$set": filtered_user_data},  # Only update provided fields
                upsert=True  )

            if result.upserted_id:
                logger.info(f"Created new user document for email: {user_data['email']}")
            else:
                logger.info(f"Updated existing user document for email: {user_data['email']}")

            return True  

        except Exception as e:
            logger.exception("Error saving user to MongoDB")
            return False


    def save_user_to_mongodb_truck(self, user_data):
        """Insert or update user data securely with camera details, including polygonal points."""
        try:
            # Fetch existing user data
            existing_user = self.user_collection.find_one({"email": user_data["email"]})
            
            if existing_user:
                # If user exists, update cameras while preserving existing rtsp_link and adding polygonal_points
                updated_cameras = existing_user.get("cameras", [])

                new_cameras = user_data.get("cameras", [])
                for new_camera in new_cameras:
                    existing_camera = next((cam for cam in updated_cameras if cam["camera_id"] == new_camera["camera_id"]), None)
                    
                    if existing_camera:
                        # Ensure polygonal_points is stored as a proper list
                        existing_camera["polygonal_points"] = new_camera.get("polygonal_points", [])
                    else:
                        # Add new camera with proper polygonal_points format
                        updated_cameras.append({
                            "camera_id": new_camera["camera_id"],
                            "rtsp_link": new_camera.get("rtsp_link", ""),
                            "polygonal_points": new_camera.get("polygonal_points", [])
                        })

                filtered_user_data = {"cameras": updated_cameras}

            else:
                # If new user, ensure required fields exist
                filtered_user_data = {
                    "name": user_data.get("name", "Unknown"),
                    "email": user_data["email"],
                    "password": self.hash_password(user_data.get("password", "defaultpassword")),  
                    "phone_no": user_data.get("phone_no", ""),
                    "role": user_data.get("role", "user"),
                    "cameras": user_data.get("cameras", [])
                }

            result = self.user_collection.update_one(
                {"email": user_data["email"]},  # Match existing user by email
                {"$set": filtered_user_data},  # Only update provided fields
                upsert=True  # Insert if it doesn't exist
            )

            if result.upserted_id:
                logger.info(f"Created new user document for email: {user_data['email']}")
            else:
                logger.info(f"Updated existing user document for email: {user_data['email']}")

            return True  # Indicate success

        except Exception as e:
            logger.exception("Error saving user to MongoDB")
            return False

        
    def fetch_camera_rtsp_by_email(self, email):
        """Retrieve camera ID and RTSP links for a given email."""
        try:
            logger.info(f"Fetching camera details for email: {email}")

            user_data = self.user_collection.find_one(
                {"email": email}, 
                {"_id": 0, "cameras": 1}  # Fetch only cameras field
            )

            if user_data and "cameras" in user_data:
                logger.info(f"Camera details found for email: {email}: {user_data['cameras']}")
                return user_data["cameras"]
            else:
                logger.warning(f"No camera details found for email: {email}")
                return None

        except Exception as e:
            logger.exception("Error fetching camera details from MongoDB")
            return None


    def has_polygonal_points(self, email):
        """Check if any camera associated with the given email has polygonal points."""
        try:
            logger.info(f"Checking for polygonal points in cameras for email: {email}")

            # Fetch the user's camera data
            user_data = self.user_collection.find_one(
                {"email": email},
                {"_id": 0, "cameras": 1}  # Only fetch the cameras field
            )

            if user_data and "cameras" in user_data:
                for camera in user_data["cameras"]:
                    if "polygonal_points" in camera and camera["polygonal_points"]:
                        logger.info(f"Polygonal points found for email: {email}")
                        return True  # Found a camera with polygonal points

            logger.info(f"No polygonal points found for email: {email}")
            return False  # No cameras have polygonal points

        except Exception as e:
            logger.exception("Error checking polygonal points")
            return False




####################################################################################################################
                                                ## Normal User ##
####################################################################################################################



# user_data = {
#     "name": "vnp",
#     "email":"vnp@gmail.com",
#     "password": "vnp123",
#     "phone_no": "1234567890",
#     "role": "admin",
#     "cameras": [
#         {"camera_id": "CAMERA_1", "rtsp_link": "Videos/07112024 2300.mp4"},
#         {"camera_id": "CAMERA_2", "rtsp_link": "Videos/09112024 0300.mp4"},
#     ]
# }


####################################################################################################################
                                                ## Loading and Unloading User ##
####################################################################################################################



# mongo_handler = MongoDBHandlerSaving()
# user_data = {
#     "name": "q",
#     "email": "q.com",
#     "password": "q.com",
#     "phone_no": "1234567890",
#     "role": "admin",
    # "cameras": [
    #     {
    #         "camera_id": "CAMERA_1",
    #         "rtsp_link": "Videos/07112024 2300.mp4",
    #         "polygonal_points": "[(571, 716), (825, 577), (1259, 616), (1256, 798)]"
    #     },
    #     {
    #         "camera_id": "CAMERA_2",
    #         "rtsp_link": "Videos/09112024 0300.mp4",
    #         "polygonal_points": "[(456, 6), (85, 57), (129, 616), (156, 798)]"
    #     }
    # ]
# }

# mongo_handler.save_user_to_mongodb(user_data)
# camera_details = mongo_handler.fetch_camera_rtsp_by_email("johndoe@example.com")
# print("Camera Details:", camera_details)



####################################################################################################################
                                                ## END ##
####################################################################################################################




