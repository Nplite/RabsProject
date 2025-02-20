
import logging
from pymongo import MongoClient
from bcrypt import hashpw, gensalt

# Configure logging
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
                    # "password":user_data["password"],

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



# mongo_handler = MongoDBHandlerSaving()
# user_data = {
#     "name": "vp",
#     "email":"vp@gmail.com",
#     "password": "vp123",
#     "phone_no": "1234567890",
#     "role": "admin",
#     "cameras": [
        # {"camera_id": "CAMERA_1", "rtsp_link": "Videos/07112024 2300.mp4"},
        # {"camera_id": "CAMERA_2", "rtsp_link": "Videos/09112024 0300.mp4"},



#     ]
# }

# mongo_handler.save_user_to_mongodb(user_data)


# camera_details = mongo_handler.fetch_camera_rtsp_by_email("johndoe@example.com")
# print("Camera Details:", camera_details)
