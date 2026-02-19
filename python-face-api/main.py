"""
AURA Face Recognition API - dlib-based (face_recognition library)
Replaces: DeepFace + TensorFlow with lightweight dlib
Optimized for: Render free tier (512MB RAM, CPU-only)
"""

import os
import logging
from datetime import datetime
from typing import List, Optional
import numpy as np
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from PIL import Image
import face_recognition
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
CORS_ORIGIN = os.getenv("CORS_ORIGIN", "https://aura-eight-beta.vercel.app")
DATABASE_NAME = "attendance_db"
COLLECTION_NAME = "face_embeddings"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# MONGODB CONNECTION
# ============================================================================

try:
    mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    # Verify connection
    mongo_client.admin.command('ping')
    db = mongo_client[DATABASE_NAME]
    embeddings_collection = db[COLLECTION_NAME]
    logger.info("✅ MongoDB connected successfully")
    DB_CONNECTED = True
except ConnectionFailure as e:
    logger.error(f"❌ MongoDB connection failed: {e}")
    embeddings_collection = None
    DB_CONNECTED = False

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="AURA Face Recognition API",
    version="5.0",
    description="dlib-based face recognition (face_recognition library)"
)

# ============================================================================
# CORS MIDDLEWARE
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://aura-eight-beta.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

@app.options("/{path:path}")
async def preflight_handler(path: str, request: Request):
    """Explicit OPTIONS handler for CORS preflight"""
    return {"detail": "OK"}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """Load image from bytes and convert to RGB numpy array"""
    try:
        img = Image.open(BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
    except Exception as e:
        logger.debug(f"Failed to load image: {e}")
        return None


def resize_image_aspect_ratio(image: np.ndarray, max_width: int = 640, 
                               max_height: int = 480) -> np.ndarray:
    """Resize image while preserving aspect ratio (max 640x480)"""
    h, w = image.shape[:2]
    if w <= max_width and h <= max_height:
        return image
    
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    img_pil = Image.fromarray(image)
    img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.array(img_pil)


def normalize_embedding(embedding: np.ndarray) -> List[float]:
    """L2 normalize embedding to unit length"""
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return (embedding / norm).tolist()
    return embedding.tolist()


def compute_average_embedding(embeddings: List[np.ndarray]) -> List[float]:
    """Compute mean embedding and L2 normalize"""
    avg = np.mean(embeddings, axis=0)
    return normalize_embedding(avg)


def check_db_connection() -> bool:
    """Check if MongoDB is connected"""
    global DB_CONNECTED, mongo_client, db, embeddings_collection
    
    if not DB_CONNECTED:
        try:
            mongo_client.admin.command('ping')
            db = mongo_client[DATABASE_NAME]
            embeddings_collection = db[COLLECTION_NAME]
            DB_CONNECTED = True
            return True
        except:
            return False
    return True





# ============================================================================
# DATA MODELS
# ============================================================================

class TrainResponse(BaseModel):
    success: bool
    student_id: str
    images_processed: int
    embeddings_stored: int
    message: Optional[str] = None


class RecognizeResponse(BaseModel):
    recognized: bool
    student_id: Optional[str] = None
    confidence: Optional[float] = None
    distance: Optional[float] = None
    reason: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    model: str
    db: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
async def health() -> HealthResponse:
    """Health check endpoint"""
    db_status = "connected" if check_db_connection() else "error"
    return HealthResponse(
        status="ok",
        service="face-recognition-api",
        model="dlib-hog-128d",
        db=db_status
    )


@app.get("/wakeup")
async def wakeup():
    """Wakeup endpoint for Render cold start"""
    logger.info("⏰ Wakeup ping received")
    return {"awake": True}


@app.post("/train")
async def train(
    student_id: str = Form(...),
    files: List[UploadFile] = File(...)
) -> TrainResponse:
    """
    Train: Extract face embeddings from images and store in MongoDB
    Expects: form with student_id (string) and files (list of images, max 10)
    """
    try:
        if not check_db_connection():
            raise HTTPException(status_code=503, detail="MongoDB not available")
        
        logger.info(f"🎓 Training student {student_id} with {len(files)} files")
        
        if len(files) < 5:
            raise HTTPException(
                status_code=400,
                detail=f"Minimum 5 images required, got {len(files)}"
            )
        
        # Limit to first 10 files
        files_to_process = files[:10]
        
        all_embeddings = []
        processed_count = 0
        
        for idx, file in enumerate(files_to_process):
            try:
                # Read and load image
                image_bytes = await file.read()
                image = load_image_from_bytes(image_bytes)
                
                if image is None:
                    logger.debug(f"Image {idx+1}: Failed to load")
                    continue
                
                # Resize for speed
                image = resize_image_aspect_ratio(image)
                
                # Detect faces (HOG model: CPU-friendly)
                face_locations = face_recognition.face_locations(
                    image, 
                    model="hog"
                )
                
                if not face_locations:
                    logger.debug(f"Image {idx+1}: No face detected")
                    continue
                
                # Get encoding for first face
                face_encodings = face_recognition.face_encodings(
                    image,
                    face_locations
                )
                
                if not face_encodings:
                    logger.debug(f"Image {idx+1}: Failed to get encoding")
                    continue
                
                # Use first face from this image
                embedding = face_encodings[0]
                all_embeddings.append(embedding)
                processed_count += 1
                logger.debug(f"Image {idx+1}: ✅ Processed (128-d embedding)")
                
            except Exception as e:
                logger.debug(f"Image {idx+1} error: {str(e)[:50]}")
                continue
        
        # Validate we have enough embeddings
        if len(all_embeddings) < 3:
            logger.warning(f"Only {len(all_embeddings)} valid faces from {len(files_to_process)} images")
            raise HTTPException(
                status_code=400,
                detail="Not enough clear face images. Please retake photos in better lighting."
            )
        
        # Compute average embedding
        avg_embedding = compute_average_embedding(all_embeddings)
        
        # Save to MongoDB (upsert)
        document = {
            "student_id": student_id,
            "embeddings": [emb.tolist() for emb in all_embeddings],
            "avg_embedding": avg_embedding,
            "image_count": len(all_embeddings),
            "updated_at": datetime.utcnow()
        }
        
        embeddings_collection.update_one(
            {"student_id": student_id},
            {"$set": document},
            upsert=True
        )
        
        logger.info(f"✅ Training complete: {len(all_embeddings)} embeddings stored for {student_id}")
        
        return TrainResponse(
            success=True,
            student_id=student_id,
            images_processed=processed_count,
            embeddings_stored=len(all_embeddings)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)) -> RecognizeResponse:
    """
    Recognize: Compare face in image against all trained students
    Expects: form with single image file called 'file'
    """
    try:
        if not check_db_connection():
            raise HTTPException(status_code=503, detail="MongoDB not available")
        
        logger.info("🔍 Recognition request")
        
        # Read and load image
        image_bytes = await file.read()
        image = load_image_from_bytes(image_bytes)
        
        if image is None:
            logger.warning("Failed to load image")
            return RecognizeResponse(
                recognized=False,
                reason="invalid_image"
            )
        
        # Resize for speed
        image = resize_image_aspect_ratio(image)
        
        # Detect faces
        face_locations = face_recognition.face_locations(image, model="hog")
        
        if not face_locations:
            logger.debug("No face detected in image")
            return RecognizeResponse(
                recognized=False,
                reason="no_face"
            )
        
        # Get encoding for first face
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if not face_encodings:
            logger.debug("Failed to get encoding")
            return RecognizeResponse(
                recognized=False,
                reason="no_encoding"
            )
        
        unknown_encoding = face_encodings[0]
        
        # Load all trained students from MongoDB
        all_students = list(embeddings_collection.find({}))
        
        if not all_students:
            logger.debug("No students trained yet")
            return RecognizeResponse(
                recognized=False,
                reason="no_students_trained"
            )
        
        logger.info(f"Comparing against {len(all_students)} trained students")
        
        # Compare against all students
        best_match_id = None
        best_distance = float('inf')
        best_confidence = 0.0
        
        for student_doc in all_students:
            student_id = student_doc["student_id"]
            avg_embedding = np.array(student_doc["avg_embedding"])
            
            # Compute distance
            distance = face_recognition.face_distance([avg_embedding], unknown_encoding)[0]
            
            # Check match (tolerance=0.5 is default)
            matches = face_recognition.compare_faces(
                [avg_embedding],
                unknown_encoding,
                tolerance=0.5
            )
            
            is_match = matches[0]
            
            logger.debug(f"  {student_id}: distance={distance:.3f}, match={is_match}")
            
            # Update best match if this is better (lower distance)
            if distance < best_distance:
                best_distance = distance
                best_match_id = student_id if is_match else None
                best_confidence = 1.0 - distance  # Convert distance to confidence
        
        # Return result
        if best_match_id is not None and best_distance <= 0.5:
            logger.info(f"✅ MATCHED: {best_match_id} (distance={best_distance:.3f})")
            return RecognizeResponse(
                recognized=True,
                student_id=best_match_id,
                confidence=float(best_confidence),
                distance=float(best_distance)
            )
        else:
            logger.info(f"❌ NO MATCH: best distance={best_distance:.3f}")
            return RecognizeResponse(
                recognized=False,
                reason="no_match",
                distance=float(best_distance)
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Recognition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/student/{student_id}")
async def delete_student(student_id: str):
    """Delete a student's embeddings from MongoDB"""
    try:
        if not check_db_connection():
            raise HTTPException(status_code=503, detail="MongoDB not available")
        
        logger.info(f"🗑️ Deleting student {student_id}")
        
        result = embeddings_collection.delete_one({"student_id": student_id})
        
        if result.deleted_count == 0:
            logger.warning(f"Student {student_id} not found")
            raise HTTPException(status_code=404, detail=f"Student {student_id} not found")
        
        logger.info(f"✅ Deleted {student_id}")
        return {"success": True, "student_id": student_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/students")
async def list_students():
    """Get list of all trained student IDs"""
    try:
        if not check_db_connection():
            raise HTTPException(status_code=503, detail="MongoDB not available")
        
        students = list(embeddings_collection.find({}, {"student_id": 1}))
        student_ids = [s["student_id"] for s in students]
        
        logger.info(f"Retrieved {len(student_ids)} trained students")
        
        return {
            "trained_students": student_ids,
            "count": len(student_ids)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ List students error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Starting AURA Face Recognition API v5.0 - dlib Edition")
    print("="*70)
    print(f"📍 Service: Face recognition using face_recognition (dlib)")
    print(f"🎯 Model: HOG face detection + 128-d dlib CNN encoding")
    print(f"💾 Database: MongoDB at {MONGODB_URI[:50]}...")
    print(f"🌐 CORS Origins: {CORS_ORIGIN}")
    print("="*70 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        timeout_keep_alive=120
    )


