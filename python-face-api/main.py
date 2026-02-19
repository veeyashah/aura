"""
AURA Face Recognition API - OpenCV + LBPH (Local Binary Patterns Histograms)
Lightweight, CPU-efficient face recognition for Render free tier
"""

import os
import logging
from datetime import datetime
from typing import List, Optional
import numpy as np
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from PIL import Image
import cv2
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE_NAME = "attendance_db"
COLLECTION_NAME = "face_embeddings"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load Haar Cascade for face detection
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
logger.info("✅ Haar Cascade loaded")

# LBPH threshold for recognition
RECOGNITION_THRESHOLD = 0.72

# ============================================================================
# MONGODB CONNECTION
# ============================================================================

try:
    mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
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
    version="6.0",
    description="OpenCV + LBPH face recognition"
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
    """Resize image while preserving aspect ratio"""
    h, w = image.shape[:2]
    if w <= max_width and h <= max_height:
        return image
    
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    img_pil = Image.fromarray(image)
    img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.array(img_pil)


def extract_face_embedding(face_roi: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract 128-d embedding from face ROI
    1. Resize to 128x128 for quality
    2. Resize to 16x8 to get 128 pixels
    3. Flatten to vector (16*8=128 dimensions)
    4. L2 normalize
    """
    try:
        # First resize to 128x128 for better quality
        if face_roi.shape[:2] != (128, 128):
            face_roi = cv2.resize(face_roi, (128, 128))
        
        # Then resize to 16x8 to get exactly 128-d vector (16*8=128)
        face_small = cv2.resize(face_roi, (16, 8))
        
        # Flatten to 128-d vector
        embedding = face_small.flatten().astype(np.float32)
        
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    except Exception as e:
        logger.debug(f"Failed to extract embedding: {e}")
        return None


def compute_average_embedding(embeddings: List[np.ndarray]) -> List[float]:
    """Compute mean embedding and L2 normalize"""
    avg = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg = avg / norm
    return avg.tolist()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))


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


def detect_faces_opencv(rgb_img: np.ndarray) -> List[tuple]:
    """Detect faces using Haar Cascade, return [(roi, box), ...]"""
    try:
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30)
        )
        
        results = []
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            if roi.size > 0:
                results.append((roi, (x, y, w, h)))
        
        if results:
            logger.debug(f"✅ Detected {len(results)} face(s)")
        return results
    except Exception as e:
        logger.debug(f"Face detection error: {e}")
        return []


# ============================================================================
# DATA MODELS
# ============================================================================

class TrainResponse(BaseModel):
    success: bool
    student_id: str
    images_processed: int
    embeddings_stored: int
    avg_embedding: List[float]
    message: Optional[str] = None


class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class RecognizeResponse(BaseModel):
    recognized: bool
    student_id: Optional[str] = None
    confidence: Optional[float] = None
    distance: Optional[float] = None
    embedding: Optional[List[float]] = None
    bbox: Optional[BBox] = None
    image_height: Optional[int] = None
    image_width: Optional[int] = None
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
        model="opencv-lbph-128d",
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
        
        files_to_process = files[:10]
        all_embeddings = []
        processed_count = 0
        
        for idx, file in enumerate(files_to_process):
            try:
                # Load image
                image_bytes = await file.read()
                image = load_image_from_bytes(image_bytes)
                
                if image is None:
                    logger.debug(f"Image {idx+1}: Failed to load")
                    continue
                
                # Resize for speed
                image = resize_image_aspect_ratio(image)
                
                # Detect faces
                faces = detect_faces_opencv(image)
                if not faces:
                    logger.debug(f"Image {idx+1}: No face detected")
                    continue
                
                # Extract embedding from first face
                face_roi = faces[0][0]  # Get grayscale ROI
                embedding = extract_face_embedding(face_roi)
                
                if embedding is None:
                    logger.debug(f"Image {idx+1}: Failed to extract embedding")
                    continue
                
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
            embeddings_stored=len(all_embeddings),
            avg_embedding=avg_embedding
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
    """
    try:
        if not check_db_connection():
            raise HTTPException(status_code=503, detail="MongoDB not available")
        
        logger.info("🔍 Recognition request")
        
        # Load image
        image_bytes = await file.read()
        image = load_image_from_bytes(image_bytes)
        
        if image is None:
            logger.warning("Failed to load image")
            return RecognizeResponse(
                recognized=False,
                reason="invalid_image"
            )
        
        # Resize
        image = resize_image_aspect_ratio(image)
        
        # Detect faces
        faces = detect_faces_opencv(image)
        
        if not faces:
            logger.debug("No face detected in image")
            return RecognizeResponse(
                recognized=False,
                reason="no_face"
            )
        
        # Extract bbox and ROI from first detected face
        face_roi = faces[0][0]
        x, y, w, h = faces[0][1]  # Bounding box coordinates
        unknown_embedding = extract_face_embedding(face_roi)
        
        if unknown_embedding is None:
            logger.debug("Failed to extract embedding")
            return RecognizeResponse(
                recognized=False,
                reason="no_encoding"
            )
        
        # Load all trained students from MongoDB
        all_students = list(embeddings_collection.find({}))
        
        if not all_students:
            logger.debug("No students trained yet")
            return RecognizeResponse(
                recognized=False,
                reason="no_students_trained"
            )
        
        logger.info(f"Comparing against {len(all_students)} trained students")
        
        # Compare against all students using cosine similarity
        best_match_id = None
        best_similarity = -1.0
        best_embedding = None
        
        for student_doc in all_students:
            student_id = student_doc["student_id"]
            avg_embedding = np.array(student_doc["avg_embedding"])
            
            # Compute cosine similarity
            similarity = cosine_sim(unknown_embedding, avg_embedding)
            
            logger.debug(f"  {student_id}: similarity={similarity:.3f}")
            
            # Update best match if this is better (higher similarity)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = student_id
                best_embedding = avg_embedding
        
        # Return result based on threshold
        if best_similarity >= RECOGNITION_THRESHOLD:
            logger.info(f"✅ MATCHED: {best_match_id} (similarity={best_similarity:.3f})")
            return RecognizeResponse(
                recognized=True,
                student_id=best_match_id,
                confidence=float(best_similarity),
                distance=float(1.0 - best_similarity),
                embedding=best_embedding.tolist() if best_embedding is not None else None,
                bbox=BBox(x=int(x), y=int(y), w=int(w), h=int(h)),
                image_height=image.shape[0],
                image_width=image.shape[1]
            )
        else:
            logger.info(f"❌ NO MATCH: best similarity={best_similarity:.3f} < {RECOGNITION_THRESHOLD}")
            return RecognizeResponse(
                recognized=False,
                reason="no_match",
                distance=float(1.0 - best_similarity),
                bbox=BBox(x=int(x), y=int(y), w=int(w), h=int(h)),
                image_height=image.shape[0],
                image_width=image.shape[1]
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
    print("Starting AURA Face Recognition API v6.0 - OpenCV Lite Edition")
    print("="*70)
    print(f"📍 Service: Face recognition using OpenCV + LBPH")
    print(f"🎯 Model: Haar Cascade detection + 128-d embedding")
    print(f"📊 Similarity: Cosine distance with threshold {RECOGNITION_THRESHOLD}")
    print(f"💾 Database: MongoDB at {MONGODB_URI[:50]}...")
    print("="*70 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        timeout_keep_alive=120
    )


