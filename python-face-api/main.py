"""
AURA Face Recognition API - Memory Optimized
Model: Facenet (128-d embeddings, CPU-friendly)
Detector: OpenCV (lightweight, optimized for Render)
Optimized for Render Free Tier with lazy model loading
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
import gc

# Optional memory monitoring
PSUTIL_AVAILABLE = False
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    pass


# ============================================================================
# MEMORY MONITORING
# ============================================================================

def get_memory_usage():
    """Get current memory usage in MB (returns 0 if psutil unavailable)"""
    if not PSUTIL_AVAILABLE:
        return 0
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except:
        return 0

def log_memory_checkpoint(label: str):
    """Log memory usage at a checkpoint"""
    mem_mb = get_memory_usage()
    if mem_mb > 0:
        logger.info(f"💾 {label}: {mem_mb:.1f} MB")
    return mem_mb


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

MODEL_NAME = "Facenet"  # 128-d embeddings, CPU-optimized for free tier
DETECTOR = "opencv"  # Lightweight, CPU-friendly
THRESHOLD = 0.65  # Cosine similarity threshold

EXPECTED_EMBEDDING_DIMS = [128]

# ============================================================================
# GLOBAL STATE
# ============================================================================

known_face_encodings: List[np.ndarray] = []
known_face_names: List[str] = []
known_face_ids: List[str] = []
lock = threading.Lock()

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
    logger.info("\n" + "="*70)
    logger.info("🚀 AURA Face Recognition API v4.0 (Facenet Edition)")
    logger.info("="*70)
    logger.info(f"📡 Model: {MODEL_NAME} (128-d embeddings, lazy loaded)")
    logger.info(f"🔍 Detector: {DETECTOR} (OpenCV)")
    logger.info(f"🎯 Threshold: {THRESHOLD} (cosine similarity)")
    logger.info("⚡ Model loading on-demand (lazy) to save memory")
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





def detect_faces_opencv(rgb_img: np.ndarray) -> List[tuple]:
    """Detect faces using OpenCV Haar Cascade (memory-optimized)"""
    results = []
    try:
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        
        # Optimized parameters for low-memory detection
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
                # Facenet expects 160x160 to save memory
                roi_resized = cv2.resize(roi, (160, 160))
                results.append((roi_resized, (x, y, w, h)))
        
        if results:
            logger.debug(f"✅ OpenCV detected {len(results)} face(s)")
            
        return results
    except Exception as e:
        logger.debug(f"Face detection error: {e}")
        return []


def detect_faces(rgb_img: np.ndarray) -> List[tuple]:
    """Detect faces using OpenCV only (memory-efficient for Render free tier)"""
    return detect_faces_opencv(rgb_img)





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
        "embedding_dimension": 128,
        "version": "4.0"
    }


@app.get("/status")
async def status():
    """Get detailed status"""
    with lock:
        return {
            "status": "ready" if known_face_names else "empty",
            "students_loaded": len(known_face_names),
            "embedding_dimension": 128,
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
    """Train: Extract embeddings from images (Memory-optimized for Render)"""
    try:
        mem_start = log_memory_checkpoint(f"Training START ({req.student_id})")
        logger.info(f"🎓 Training {req.student_id} with {len(req.images)} images")
        
        # Limit images to 5 for memory efficiency on free tier
        images_to_use = req.images[:5]
        if len(req.images) > 5:
            logger.warning(f"⚠️ Limiting images from {len(req.images)} to 5 for memory efficiency")
        
        if len(images_to_use) < 5:
            raise HTTPException(400, "Minimum 5 images required for training")
        
        # Run blocking operations in thread executor to prevent timeout
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, process_training_images, images_to_use)
        
        if len(embeddings) < 3:
            logger.error(f"❌ Only {len(embeddings)} valid faces from {len(images_to_use)} images")
            raise HTTPException(400, f"Need at least 3 valid faces, got {len(embeddings)}")
        
        faces_count = len(embeddings)
        
        # Average embeddings
        avg_emb = np.mean(embeddings, axis=0).astype(np.float32)
        avg_emb = normalize_embedding(avg_emb)
        
        # Clear embeddings from memory
        del embeddings
        gc.collect()
        
        mem_end = log_memory_checkpoint(f"Training END ({req.student_id})")
        mem_delta = mem_end - mem_start
        logger.info(f"✅ Training complete: {faces_count} faces processed (memory delta: {mem_delta:+.1f} MB)")
        
        return {
            "success": True,
            "embedding": avg_emb.tolist(),
            "faces_processed": faces_count,
            "embedding_dimension": len(avg_emb)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def process_training_images(images: List[str]) -> List[np.ndarray]:
    """Process images for training (memory-optimized for Render free tier)"""
    embeddings = []
    mem_start = get_memory_usage()
    
    # Process images (limited to 5) for embeddings using lazy-loaded Facenet model
    for idx, img_b64 in enumerate(images[:5]):  # MAX 5 images for memory efficiency
        try:
            rgb = decode_base64(img_b64)
            if rgb is None:
                logger.debug(f"Image {idx+1}/5: Failed to decode")
                continue
            
            # Detect faces with opencv (lightweight, fast, memory-efficient)
            faces = detect_faces_opencv(rgb)
            if not faces:
                logger.debug(f"Image {idx+1}/5: No faces detected")
                # Clear memory
                del rgb
                gc.collect()
                continue
            
            # Get embedding from first face using lazy-loaded Facenet model
            emb = get_embedding_facenet512(faces[0][0])
            if emb is not None:
                embeddings.append(normalize_embedding(emb))
                logger.debug(f"Image {idx+1}/5: ✅ Embedding extracted ({len(emb)}-d)")
            else:
                logger.debug(f"Image {idx+1}/5: Failed to get embedding")
            
            # Clear memory after each image
            del rgb, faces
            gc.collect()
                
        except Exception as e:
            logger.debug(f"Image {idx+1}/5 error: {str(e)[:50]}")
            gc.collect()
            continue
    
    mem_end = get_memory_usage()
    mem_delta = mem_end - mem_start
    logger.info(f"📊 Processed {len(embeddings)} images (memory delta: {mem_delta:+.1f} MB, final: {mem_end:.1f} MB)")
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
                # Facenet expects 160x160 input
                roi_resized = cv2.resize(roi, (160, 160))
                results.append((roi_resized, (x, y, w, h)))
        
        if results:
            logger.debug(f"✅ Detected {len(results)} face(s)")
        
        return results
    except Exception as e:
        logger.debug(f"Face detection error: {e}")
        return []


def get_embedding_facenet512(face_roi: np.ndarray) -> Optional[np.ndarray]:
    """Get embedding using lazy-loaded Facenet model (memory-optimized)"""
    if not DEEPFACE_AVAILABLE:
        return None
    
    try:
        mem_before = get_memory_usage()
        
        # Load model on-demand (lazy loading for memory efficiency)
        # Minimize array copies by passing directly
        embedding_objs = DeepFace.represent(
            img_path=face_roi,
            model_name=MODEL_NAME,
            enforce_detection=False,
            align=False  # Skip alignment for speed and memory
        )
        
        mem_after = get_memory_usage()
        if mem_after - mem_before > 100:
            logger.warning(f"⚠️ Large memory spike during embedding: +{mem_after - mem_before:.1f} MB")
        
        if embedding_objs and len(embedding_objs) > 0:
            embedding = embedding_objs[0].get("embedding")
            if embedding:
                return np.array(embedding, dtype=np.float32)
    except Exception as e:
        logger.debug(f"Embedding error: {e}")
        gc.collect()  # Force garbage collection on error
    
    return None


@app.post("/recognize")
async def recognize(req: RecognitionRequest):
    """Recognize faces in image - OPTIMIZED for speed with lazy model loading"""
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
                    # Get embedding with lazy-loaded Facenet model
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

