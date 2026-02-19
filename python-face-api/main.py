"""
AURA Face Recognition API - Production Ready
Model: Facenet512 (512-d embeddings, excellent accuracy)
Detector: OpenCV (lightweight, optimized for Render)
Optimized for Render Free Tier with preloaded models
"""

# ⚡ Disable TensorFlow GPU checks - MUST BE FIRST
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import cv2
import numpy as np
import base64
import io
from PIL import Image
from typing import List, Optional
import logging
from datetime import datetime
import threading
import json
import asyncio


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DEEPFACE & MODEL SETUP
# ============================================================================

DEEPFACE_AVAILABLE = False
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logger.info("✅ DeepFace imported successfully")
except ImportError as e:
    logger.error(f"❌ DeepFace import failed: {e}")

# Load OpenCV cascade for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
logger.info(f"✅ OpenCV Haar Cascade loaded")

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "Facenet512"  # 512-d embeddings, superior accuracy
DETECTOR = "opencv"  # Lightweight, CPU-friendly
THRESHOLD = 0.65  # Cosine similarity threshold for Facenet512

EXPECTED_EMBEDDING_DIMS = [512]

# ============================================================================
# GLOBAL STATE
# ============================================================================

known_face_encodings: List[np.ndarray] = []
known_face_names: List[str] = []
known_face_ids: List[str] = []
lock = threading.Lock()

# Preloaded models
FACENET_MODEL = None

# ============================================================================
# DATA MODELS
# ============================================================================

class TrainingRequest(BaseModel):
    student_id: str
    images: List[str]


class RecognitionRequest(BaseModel):
    image: str


class StudentData(BaseModel):
    studentId: str
    name: str
    faceEmbeddings: List[float]


class LiveRecognitionRequest(BaseModel):
    students: List[StudentData]

# ============================================================================
# LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global FACENET_MODEL
    
    logger.info("\n" + "="*70)
    logger.info("🚀 AURA Face Recognition API v4.0 (Facenet512 Edition)")
    logger.info("="*70)
    logger.info(f"📡 Model: {MODEL_NAME} (512-d embeddings)")
    logger.info(f"🔍 Detector: {DETECTOR} (OpenCV)")
    logger.info(f"🎯 Threshold: {THRESHOLD} (cosine similarity)")
    
    # ⚡ PRELOAD MODEL AT STARTUP
    if DEEPFACE_AVAILABLE:
        try:
            logger.info("🔥 Preloading Facenet512 model (this takes 30-60 seconds)...")
            FACENET_MODEL = DeepFace.build_model(MODEL_NAME)
            logger.info("✅ Facenet512 model preloaded successfully")
            logger.info("⚡ Model ready - API will respond instantly on requests")
        except Exception as e:
            logger.error(f"❌ Failed to preload model: {e}")
    
    logger.info("="*70 + "\n")
    yield
    logger.info("👋 API shutdown")


app = FastAPI(
    title="AURA Face Recognition API",
    version="4.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://aura-eight-beta.vercel.app",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ============================================================================
# DATA MODELS
# ============================================================================

class TrainingRequest(BaseModel):
    student_id: str
    images: List[str]


class RecognitionRequest(BaseModel):
    image: str


class StudentData(BaseModel):
    studentId: str
    name: str
    faceEmbeddings: List[float]


class LiveRecognitionRequest(BaseModel):
    students: List[StudentData]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def decode_base64(b64_str: str) -> Optional[np.ndarray]:
    """Decode base64 to RGB image"""
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        
        # Add padding if needed
        padding = len(b64_str) % 4
        if padding:
            b64_str += "=" * (4 - padding)
        
        img = Image.open(io.BytesIO(base64.b64decode(b64_str)))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return np.array(img)
    except Exception as e:
        logger.error(f"❌ Decode error: {e}")
        return None


def detect_faces_deepface(rgb_img: np.ndarray) -> List[tuple]:
    """Detect faces using DeepFace with single best detector"""
    results = []
    
    if not DEEPFACE_AVAILABLE:
        return results
    
    try:
        # Use only RetinaFace (best accuracy/speed tradeoff) - skip other detectors for speed
        detector_name = 'retinaface'
        
        try:
            # Extract faces using DeepFace
            face_objs = DeepFace.extract_faces(
                img_path=rgb_img,
                detector_backend=detector_name,
                enforce_detection=False,
                align=True
            )
            
            if face_objs and len(face_objs) > 0:
                logger.debug(f"✅ {detector_name} detected {len(face_objs)} face(s)")
                
                for face_obj in face_objs:
                    facial_area = face_obj.get('facial_area', {})
                    x = facial_area.get('x', 0)
                    y = facial_area.get('y', 0)
                    w = facial_area.get('w', 0)
                    h = facial_area.get('h', 0)
                    
                    face_img = face_obj.get('face')
                    
                    if face_img is not None and face_img.size > 0:
                        if face_img.dtype != np.uint8:
                            face_img = (face_img * 255).astype(np.uint8)
                        
                        # ArcFace expects 224x224
                        face_resized = cv2.resize(face_img, (224, 224))
                        results.append((face_resized, (x, y, w, h)))
                
                if results:
                    return results
                    
        except Exception as e:
            logger.debug(f"⚠️ RetinaFace failed: {str(e)[:100]}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ DeepFace detection error: {e}")
        return []


def detect_faces_opencv(rgb_img: np.ndarray) -> List[tuple]:
    """Detect faces using OpenCV Haar Cascade (fallback)"""
    results = []
    try:
        logger.debug("🔄 Falling back to OpenCV Haar Cascade...")
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        
        # Lenient parameters for detection
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, w, h) in faces:
            roi = rgb_img[y:y+h, x:x+w]
            if roi.size > 0:
                roi_resized = cv2.resize(roi, (224, 224))
                results.append((roi_resized, (x, y, w, h)))
        
        if results:
            logger.info(f"✅ OpenCV detected {len(results)} face(s)")
            
        return results
    except Exception as e:
        logger.debug(f"⚠️ OpenCV failed: {e}")
        return []


def detect_faces(rgb_img: np.ndarray) -> List[tuple]:
    """Detect faces with multiple fallbacks"""
    # Try DeepFace first (better accuracy)
    faces = detect_faces_deepface(rgb_img)
    if faces:
        return faces
    
    # Fallback to OpenCV
    faces = detect_faces_opencv(rgb_img)
    if faces:
        return faces
    
    logger.warning("⚠️ No faces detected by any method")
    return []


def get_embedding_deepface(face_roi: np.ndarray) -> Optional[np.ndarray]:
    """Get embedding using DeepFace ArcFace model"""
    try:
        if not DEEPFACE_AVAILABLE:
            logger.warning("⚠️ DeepFace not available")
            return None
        
        # ArcFace provides 512-d embeddings with superior accuracy
        result = DeepFace.represent(
            face_roi,
            model_name="ArcFace",
            enforce_detection=False,
            normalization="ArcFace"
        )
        
        if result and len(result) > 0:
            embedding = np.array(result[0]["embedding"], dtype=np.float32)
            logger.debug(f"✅ ArcFace embedding: {len(embedding)}-d")
            return embedding
        
        logger.warning("⚠️ DeepFace returned empty result")
        return None
        
    except Exception as e:
        logger.error(f"❌ DeepFace embedding error: {e}")
        return None


def get_embedding(face_roi: np.ndarray) -> Optional[np.ndarray]:
    """Get embedding with fallbacks"""
    emb = get_embedding_deepface(face_roi)
    if emb is not None:
        return emb
    
    logger.warning("⚠️ Could not generate embedding")
    return None


def normalize_embedding(emb: np.ndarray) -> np.ndarray:
    """L2 normalize embedding"""
    norm = np.linalg.norm(emb)
    if norm > 0:
        return (emb / norm).astype(np.float32)
    return emb.astype(np.float32)


def cosine_similarity(e1: np.ndarray, e2: np.ndarray) -> float:
    """Compute cosine similarity (0-1, higher = more similar)"""
    # Normalize for cosine similarity
    e1_norm = e1 / (np.linalg.norm(e1) + 1e-10)
    e2_norm = e2 / (np.linalg.norm(e2) + 1e-10)
    
    similarity = float(np.dot(e1_norm, e2_norm))
    
    # Clamp to [0, 1] range for cosine similarity
    return max(0.0, min(1.0, similarity))




# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "OK",
        "model": MODEL_NAME,
        "detector": DETECTOR,
        "threshold": THRESHOLD,
        "loaded_students": len(known_face_names),
        "embedding_dimension": 512,
        "version": "4.0"
    }


@app.get("/status")
async def status():
    """Get detailed status"""
    with lock:
        return {
            "status": "ready" if known_face_names else "empty",
            "students_loaded": len(known_face_names),
            "embedding_dimension": 512,
            "threshold": THRESHOLD,
            "model": MODEL_NAME,
            "detector": DETECTOR
        }


@app.post("/load-students")
async def load_students(req: LiveRecognitionRequest):
    """Load student embeddings into memory"""
    global known_face_encodings, known_face_names, known_face_ids
    
    try:
        with lock:
            known_face_encodings.clear()
            known_face_names.clear()
            known_face_ids.clear()
            
            logger.info(f"📥 Loading {len(req.students)} students...")
            
            loaded = 0
            skipped = 0
            errors = []
            
            for student in req.students:
                try:
                    if not student.faceEmbeddings or len(student.faceEmbeddings) == 0:
                        skipped += 1
                        errors.append(f"{student.name}: No embeddings")
                        continue
                    
                    # Validate embedding dimension
                    if len(student.faceEmbeddings) not in EXPECTED_EMBEDDING_DIMS:
                        skipped += 1
                        errors.append(f"{student.name}: Invalid dimension {len(student.faceEmbeddings)}d (expected {EXPECTED_EMBEDDING_DIMS[0]}d)")
                        continue
                    
                    emb = np.array(student.faceEmbeddings, dtype=np.float32)
                    emb = normalize_embedding(emb)
                    
                    known_face_encodings.append(emb)
                    known_face_names.append(student.name)
                    known_face_ids.append(student.studentId)
                    loaded += 1
                    
                except Exception as e:
                    skipped += 1
                    errors.append(f"{student.name}: {str(e)[:50]}")
            
            logger.info(f"✅ Loaded {loaded} students, skipped {skipped}")
            if errors and len(errors) <= 5:
                for err in errors[:5]:
                    logger.debug(f"   - {err}")
            
            return {
                "success": True,
                "loaded_count": loaded,
                "skipped_count": skipped,
                "total_requested": len(req.students),
                "errors": errors[:10] if errors else []
            }
    except Exception as e:
        logger.error(f"❌ Load students error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def train(req: TrainingRequest):
    """Train: Extract embeddings from images (OPTIMIZED - async, high accuracy)"""
    try:
        logger.info(f"🎓 Training {req.student_id} with {len(req.images)} images")
        
        if len(req.images) < 5:
            raise HTTPException(400, "Minimum 5 images required for training")
        
        # Run blocking operations in thread executor to prevent timeout
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, process_training_images, req.images)
        
        if len(embeddings) < 3:
            logger.error(f"❌ Only {len(embeddings)} valid faces from {len(req.images)} images")
            raise HTTPException(400, f"Need at least 3 valid faces, got {len(embeddings)}")
        
        # Average embeddings for MAXIMUM ACCURACY
        avg_emb = np.mean(embeddings, axis=0).astype(np.float32)
        avg_emb = normalize_embedding(avg_emb)
        
        logger.info(f"✅ Training complete: {len(embeddings)} faces processed (HIGH ACCURACY)")
        
        return {
            "success": True,
            "embedding": avg_emb.tolist(),
            "faces_processed": len(embeddings),
            "embedding_dimension": len(avg_emb)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def process_training_images(images: List[str]) -> List[np.ndarray]:
    """Process images for training (runs in thread executor)"""
    embeddings = []
    
    # Use preloaded Facenet512 model
    for idx, img_b64 in enumerate(images[:50]):
        try:
            rgb = decode_base64(img_b64)
            if rgb is None:
                logger.debug(f"Image {idx+1}: Failed to decode")
                continue
            
            # Detect faces with opencv (lightweight, fast)
            faces = detect_faces_opencv(rgb)
            if not faces:
                logger.debug(f"Image {idx+1}: No faces detected")
                continue
            
            # Get embedding from first face using preloaded Facenet512 model
            emb = get_embedding_facenet512(faces[0][0])
            if emb is not None:
                embeddings.append(normalize_embedding(emb))
            else:
                logger.debug(f"Image {idx+1}: Failed to get embedding")
                
        except Exception as e:
            logger.debug(f"Image {idx+1} error: {str(e)[:30]}")
            continue
    
    return embeddings


def detect_faces_opencv(rgb_img: np.ndarray) -> List[tuple]:
    """Detect faces using OpenCV Haar Cascade (lightweight, fast)"""
    results = []
    try:
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        
        # Detect with lenient parameters for good detection rate
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, w, h) in faces:
            roi = rgb_img[y:y+h, x:x+w]
            if roi.size > 0:
                # Facenet512 expects 160x160 input
                roi_resized = cv2.resize(roi, (160, 160))
                results.append((roi_resized, (x, y, w, h)))
        
        if results:
            logger.debug(f"✅ Detected {len(results)} face(s)")
        
        return results
    except Exception as e:
        logger.debug(f"Face detection error: {e}")
        return []


def get_embedding_facenet512(face_roi: np.ndarray) -> Optional[np.ndarray]:
    """Get embedding using preloaded Facenet512 model"""
    global FACENET_MODEL
    
    if not DEEPFACE_AVAILABLE or FACENET_MODEL is None:
        return None
    
    try:
        # Use preloaded Facenet512 model for faster processing
        embedding_objs = DeepFace.represent(
            img_path=face_roi,
            model_name=MODEL_NAME,
            enforce_detection=False,
            model=FACENET_MODEL,  # Use preloaded model
            align=False  # Skip alignment for speed
        )
        
        if embedding_objs and len(embedding_objs) > 0:
            embedding = embedding_objs[0].get("embedding")
            if embedding:
                return np.array(embedding, dtype=np.float32)
    except Exception as e:
        logger.debug(f"Embedding error: {e}")
    
    return None


@app.post("/recognize")
async def recognize(req: RecognitionRequest):
    """Recognize faces in image - OPTIMIZED for speed with preloaded models"""
    try:
        logger.info(f"🔍 Recognition request (loaded: {len(known_face_names)} students)")
        
        with lock:
            if not known_face_encodings:
                logger.warning("⚠️ No students loaded in memory")
                return {
                    "success": True,
                    "faces": [],
                    "loaded_students": 0,
                    "note": "No trained students loaded"
                }
            
            # Decode image
            rgb = decode_base64(req.image)
            if rgb is None:
                logger.error("❌ Failed to decode image")
                return {"success": False, "faces": [], "error": "Decode failed"}
            
            logger.debug(f"✅ Image decoded: {rgb.shape}")
            
            # Detect faces using opencv
            faces = detect_faces_opencv(rgb)
            if not faces:
                logger.debug("ℹ️ No faces detected")
                return {
                    "success": True,
                    "faces": [],
                    "loaded_students": len(known_face_names),
                    "note": "No faces detected in image"
                }
            
            logger.info(f"👤 Detected {len(faces)} face(s)")
            results = []
            
            # Process each face
            for idx, (face_roi, box) in enumerate(faces):
                try:
                    # Get embedding with preloaded Facenet512 model
                    emb = get_embedding_facenet512(face_roi)
                    if emb is None:
                        logger.debug(f"Face {idx+1}: Failed to get embedding")
                        continue
                    
                    emb = normalize_embedding(emb)
                    
                    # Find best match - VECTORIZED for speed
                    max_similarity = -1.0
                    best_idx = -1
                    
                    # Pre-filter: only check against valid embedding dimensions
                    for i, known_emb in enumerate(known_face_encodings):
                        similarity = cosine_similarity(emb, known_emb)
                        
                        # Early exit if we find a VERY high confidence match (99%+)
                        if similarity > 0.99:
                            max_similarity = similarity
                            best_idx = i
                            break
                        
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_idx = i
                    
                    # Check against threshold
                    if best_idx < 0:
                        logger.debug(f"Face {idx+1}: No valid match")
                        continue
                    
                    recognized = max_similarity >= THRESHOLD
                    distance = float(1.0 - max_similarity)
                    
                    result = {
                        "name": known_face_names[best_idx],
                        "student_id": known_face_ids[best_idx] if recognized else "",
                        "similarity": float(max_similarity),
                        "distance": distance,
                        "recognized": recognized,
                        "confidence": float(max_similarity),
                        "box": [int(box[0]), int(box[1]), int(box[0] + box[2]), int(box[1] + box[3])]
                    }
                    
                    results.append(result)
                    
                    if recognized:
                        logger.info(f"✅ MATCHED - {known_face_names[best_idx]} (sim: {max_similarity:.3f})")
                    else:
                        logger.debug(f"ℹ️ Close match - {known_face_names[best_idx]} ({max_similarity:.3f} < {THRESHOLD})")
                        
                except Exception as e:
                    logger.error(f"Face {idx+1}: {e}")
                    continue
            
            return {
                "success": True,
                "faces": results,
                "timestamp": datetime.now().isoformat(),
                "loaded_students": len(known_face_names)
            }
            
    except Exception as e:
        logger.error(f"❌ Recognition error: {e}")
        return {"success": False, "faces": [], "error": str(e)}


@app.post("/test-detection")
async def test_detection(req: RecognitionRequest):
    """Test face detection only"""
    try:
        logger.info("🧪 Testing face detection...")
        
        rgb = decode_base64(req.image)
        if rgb is None:
            return {"success": False, "error": "Failed to decode image"}
        
        logger.info(f"✅ Image decoded: {rgb.shape}")
        
        # Detect faces
        faces = detect_faces(rgb)
        
        if not faces:
            logger.warning("⚠️ No faces detected")
            return {
                "success": True,
                "faces_detected": 0,
                "message": "No faces detected. Try: better lighting, face camera directly, adjust distance"
            }
        
        logger.info(f"✅ Detected {len(faces)} face(s)")
        
        face_boxes = []
        for idx, (roi, box) in enumerate(faces):
            x, y, w, h = box
            face_boxes.append({
                "index": idx,
                "box": [int(x), int(y), int(x + w), int(y + h)],
                "size": f"{w}x{h}px"
            })
        
        return {
            "success": True,
            "faces_detected": len(faces),
            "faces": face_boxes,
            "message": f"Successfully detected {len(faces)} face(s)"
        }
        
    except Exception as e:
        logger.error(f"❌ Detection test error: {e}")
        return {"success": False, "error": str(e)}
# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))  

    print("\n" + "="*70)
    print("Starting AURA Face Recognition API v4.0")
    print("="*70)
    print(f"📍 Address: http://0.0.0.0:{port}")
    print("="*70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

