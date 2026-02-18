# DeepFace + ArcFace Setup (High Accuracy)

## What Changed

### Pure DeepFace Implementation
- **Detection:** RetinaFace → MTCNN → OpenCV (in order of accuracy)
- **Recognition:** ArcFace (512-d embeddings, best accuracy)
- **Similarity:** Cosine similarity (0-1 scale, higher = better match)
- **Threshold:** 0.68 similarity (strict, equivalent to 0.35 euclidean distance)

### Why ArcFace?
- State-of-the-art face recognition model
- 512-dimensional embeddings
- Superior accuracy compared to VGGFace/Facenet
- Robust to lighting, angles, expressions

### Detection Cascade
1. **RetinaFace** - Best accuracy, handles difficult angles/lighting
2. **MTCNN** - Good accuracy, fast
3. **OpenCV** - Fallback, very fast

## Installation

### Step 1: Install Better Detectors
```bash
install-deepface-detectors.bat
```

Or manually:
```bash
cd python-face-api
pip install retina-face mtcnn
```

### Step 2: Restart Python API
```bash
cd python-face-api
python main.py
```

You should see:
```
✅ DeepFace imported successfully
🎯 Threshold: 0.68
```

### Step 3: Retrain Students (IMPORTANT!)
Since we switched from VGGFace to ArcFace, embeddings are different.

**You MUST retrain all students:**
1. Login as admin
2. Go to Admin → Students
3. For each student:
   - Click on student
   - Click "Train Face Recognition"
   - Capture 50 images
   - Wait for training to complete

### Step 4: Test Live Recognition
1. Login as faculty
2. Go to Live Attendance
3. Start live recognition
4. Should now detect and recognize faces!

## Expected Behavior

### Python API Console
```
🔍 Recognition request received (loaded students: 4)
✅ Image decoded: (480, 640, 3)
Trying retinaface detector...
✅ retinaface detected 1 face(s)
✅ ArcFace embedding: 512-d
✅ Recognized: John Doe (similarity: 0.892)
```

### Browser Console
```
📸 Processing frame...
🔍 Sending frame to http://localhost:8000/recognize...
✅ Python API responded: 200
📊 Recognition result: {success: true, faces: [{...similarity: 0.892...}]}
👤 Found 1 face(s)
✅ John Doe marked present (d: 0.108)
```

## Detection Methods

| Detector | Accuracy | Speed | Use Case |
|----------|----------|-------|----------|
| RetinaFace | 99%+ | ~200ms | Best for difficult conditions |
| MTCNN | 95%+ | ~100ms | Good balance |
| OpenCV | 70%+ | ~10ms | Fast fallback |

The API tries them in order and uses the first one that detects faces.

## Recognition Accuracy

### ArcFace Similarity Scale
- **0.90-1.00** - Excellent match (same person, good conditions)
- **0.68-0.90** - Good match (same person, acceptable conditions)
- **0.50-0.68** - Poor match (might be same person, bad conditions)
- **0.00-0.50** - No match (different person)

**Threshold: 0.68** - Only auto-marks if similarity ≥ 0.68

### Distance (for frontend compatibility)
- Python API converts similarity to distance: `distance = 1.0 - similarity`
- **0.00-0.10** - Excellent (similarity 0.90-1.00)
- **0.10-0.32** - Good (similarity 0.68-0.90)
- **0.32-0.50** - Poor (similarity 0.50-0.68)
- **0.50-1.00** - No match (similarity 0.00-0.50)

## Troubleshooting

### Issue: Still not detecting faces

**Check Python API console:**
```
Trying retinaface detector...
retinaface failed: ...
Trying mtcnn detector...
mtcnn failed: ...
Falling back to OpenCV Haar Cascade...
✅ OpenCV detected 1 face(s)
```

If all fail, it's a lighting/positioning issue:
- **Improve lighting** - Face should be well-lit, no harsh shadows
- **Face camera directly** - Look straight at camera
- **Adjust distance** - 30-100cm from camera
- **Plain background** - Avoid busy backgrounds

### Issue: Faces detected but not recognized

**Check similarity scores in Python API console:**
```
✅ Recognized: John Doe (similarity: 0.892)  ← Good!
```

If similarity is low (< 0.68):
- **Retrain student** - Capture better quality images
- **Better lighting during training** - Well-lit, clear images
- **Consistent conditions** - Train in similar lighting to recognition

### Issue: RetinaFace/MTCNN not installing

**Windows:**
```bash
pip install --upgrade pip
pip install retina-face mtcnn
```

**If still fails:**
The system will fall back to OpenCV (still works, just less accurate).

### Issue: Training fails with ArcFace

Check Python API console during training:
```
✅ ArcFace embedding: 512-d
```

If you see errors:
- Ensure DeepFace is installed: `pip install deepface`
- Ensure TensorFlow is installed: `pip install tensorflow`
- Restart Python API

## Performance

### Detection Speed
- RetinaFace: ~200ms per frame
- MTCNN: ~100ms per frame
- OpenCV: ~10ms per frame

Most frames will use RetinaFace (best accuracy). If too slow, it falls back to faster methods.

### Recognition Speed
- ArcFace embedding: ~100ms per face
- Similarity comparison: ~1ms per student

Total: ~300ms per frame with 1 face and 10 students.

### Optimization Tips

If too slow:
1. **Increase processing interval:**
   ```javascript
   // In frontend/app/faculty/live-attendance/page.tsx
   }, 2000) // Change from 1000ms to 2000ms
   ```

2. **Reduce image quality:**
   ```javascript
   const imageData = tempCanvas.toDataURL('image/jpeg', 0.5) // Change from 0.65 to 0.5
   ```

3. **Limit students:**
   - Only load students for specific batch/year
   - Fewer students = faster comparison

## Testing

### Test 1: Check Python API
```bash
curl http://localhost:8000/health
```
Should show: `"model":"DeepFace"`

### Test 2: Test Detection
Open `test-face-detection.html`:
1. Start camera
2. Test detection
3. Should detect faces with RetinaFace/MTCNN

### Test 3: Check Embeddings
After retraining students:
```bash
node check-students.js
```
Should show: `Embeddings: 512 elements` (ArcFace)

### Test 4: Live Recognition
1. Login as faculty
2. Start live recognition
3. Should detect and recognize faces
4. Check similarity scores in Python API console

## Summary

### What You Need to Do

1. **Install detectors:**
   ```bash
   install-deepface-detectors.bat
   ```

2. **Restart Python API:**
   ```bash
   cd python-face-api
   python main.py
   ```

3. **Retrain ALL students** (CRITICAL!)
   - Old embeddings are VGGFace (incompatible)
   - New embeddings are ArcFace (512-d)
   - Must retrain for recognition to work

4. **Test live recognition**

### Expected Results

- **Detection:** RetinaFace should detect faces in most conditions
- **Recognition:** ArcFace should recognize with 95%+ accuracy
- **Threshold:** 0.68 similarity (strict, low false positives)

The system now uses state-of-the-art face recognition with DeepFace + ArcFace!
