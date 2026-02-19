# 🎓 Smart Attendance Management System

A production-ready, AI-powered attendance system with real-time face recognition.

## ✨ Features

- 🎥 **Live Face Recognition** - Real-time attendance marking using AI
- 👨‍💼 **Admin Portal** - Complete management of students, faculty, and timetables
- 👨‍🏫 **Faculty Portal** - Easy attendance marking and reporting
- 📊 **Analytics Dashboard** - Subject-wise attendance statistics
- 📥 **Export Reports** - Download attendance data in Excel format
- 🎨 **Modern UI** - Production-grade, responsive design
- 🔐 **Secure Authentication** - JWT-based auth with role-based access

## 🚀 Quick Start

### Prerequisites
- Node.js v16+
- MongoDB
- Python 3.8+
- Webcam

### Installation

# 🎓 Smart Attendance Management System

A production-ready, AI-powered attendance system with real-time face recognition.

## Highlights (updated)

- Server-side face recognition now uses **Facenet512** embeddings (512-d) and **OpenCV** detector for a CPU-friendly, production-ready pipeline.
- Models are preloaded at API startup for fast request responses on Render/VPS.
- Training remains high-quality (multiple images per student) while recognition is optimized for speed (<400ms typical per frame).

## ✨ Features

- 🎥 Live Face Recognition — Real-time attendance marking using server-side AI
- 👨‍💼 Admin Portal — Manage students, faculty, and timetables
- 👨‍🏫 Faculty Portal — Live attendance and reporting
- 📊 Analytics Dashboard — Subject-wise attendance statistics
- 📥 Export Reports — Excel/CSV export
- 🔐 Secure Authentication — JWT-based auth with role-based access

## 🚀 Quick Start (updated)

### Prerequisites
- Node.js v16+
- MongoDB (Atlas or local)
- Python 3.8+ (for `python-face-api`)
- Webcam (for training / live attendance pages)

### Installation

1. Clone the repository
   ```bash
   git clone <repository-url>
   cd AURA
   ```

2. Install dependencies
   ```bash
   # Backend
   cd backend
   npm install

   # Frontend
   cd ../frontend
   npm install

   # Python API - install into a virtualenv
   cd ../python-face-api
   python -m venv .venv
   .venv\Scripts\activate    # Windows
   # or: source .venv/bin/activate  # macOS/Linux
   pip install -r requirements.txt
   ```

3. Configure environment
- Copy `backend/.env.example` → `backend/.env` and set `CORS_ORIGIN`, `MONGODB_URI`, etc.
- Copy `frontend/.env.local.example` → `frontend/.env.local` and set `NEXT_PUBLIC_FACE_API_URL`/`NEXT_PUBLIC_BACKEND_URL`.

4. Start services (manual)
   ```bash
   # Terminal 1: Backend
   cd backend
   node server.js

   # Terminal 2: Frontend
   cd frontend
   npm run dev

   # Terminal 3: Python API
   cd python-face-api
   .venv\Scripts\activate
   python main.py
   ```

5. Access the app
- Frontend: `http://localhost:3000`

## ⚙️ Python Face API (important changes)

- Model: `Facenet512` (server-side DeepFace wrapper)
- Detector: `opencv` (OpenCV Haar cascade) — chosen for CPU efficiency on Render/free-tier
- Input size for model: 160x160 (Facenet standard)
- Threshold (cosine similarity): **0.65** (adjustable in `python-face-api/main.py`)
- Preloading: The model is loaded once during API startup to avoid per-request loading delays.
- TensorFlow: The Python API requires TensorFlow (Keras backend) for DeepFace models. The API sets env vars to disable GPU attempts and quiet logs:
  - `TF_CPP_MIN_LOG_LEVEL=3`
  - `CUDA_VISIBLE_DEVICES=-1`

Recommended: keep the TensorFlow dependency for the server. If you must avoid TF, port the pipeline to a PyTorch/ONNX alternative or use `face_recognition` (dlib), which requires code changes.

### Typical performance (observed targets)
- Training (20 images): ~45–60s on CPU (models preloaded)
- Recognition latency: <400ms per frame (preloaded model + vectorized comparison)

## 🔧 Frontend notes

- The frontend previously included `face-api.js` and `@tensorflow/tfjs` in `package.json`. Those are only required if you want client-side face detection/embedding.
- Current recommended setup: perform all detection/embedding server-side (the FastAPI Python service). Removing client-side TF reduces bundle size and complexity.

To remove client-side TensorFlow and face-api.js (optional):
```bash
cd frontend
npm remove face-api.js @tensorflow/tfjs
npm install
```

## 📊 Face Recognition Flow (updated)

1. Admin captures N images per student for training (UI defaults to 20 images; the API accepts up to 50 but recommends 20 for speed/quality).
2. Images are uploaded to `POST /train` on the Python API; embeddings are computed and averaged (L2-normalized 512-d vector stored).
3. Faculty uses Live Attendance: frontend sends camera frames to `POST /recognize` (or the backend calls the API). The Python API detects faces with OpenCV, computes embeddings (Facenet512), and compares against loaded student embeddings using cosine similarity.

## 🔐 Security & Deployment Notes

- Keep `TF_CPP_MIN_LOG_LEVEL` and `CUDA_VISIBLE_DEVICES` settings in `python-face-api/main.py` to avoid TensorFlow GPU initialization on CPU-only hosts.
- Use environment variables for production endpoints:
  - `NEXT_PUBLIC_FACE_API_URL` in `frontend/.env.local`
  - `CORS_ORIGIN` and `MONGODB_URI` in `backend/.env`

## 🧪 Testing

Test backend routes
```bash
cd backend
node test-routes.js
```

Test Python API health
```bash
curl http://localhost:8000/health
```

## 🐛 Troubleshooting (updated)

**Python API startup slow**
- Expect 20–60s for model preload on first startup; watch logs for "preloaded" messages.

**Model/TF errors on Render**
- Ensure `CUDA_VISIBLE_DEVICES=-1` is set before TensorFlow imports (this is already done at the top of `main.py`).

**Frontend bundle large**
- Remove `@tensorflow/tfjs` and `face-api.js` if you don't need client-side processing.

## 📝 Credentials (default)

- Admin
  - Email: `admin@attendance.com`
  - Password: `admin123`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit and push changes
4. Open a Pull Request

## 📄 License

MIT

---

Made with ❤️ for educational institutions

```
node test-routes.js
