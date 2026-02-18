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

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd attendance-system
   ```

2. **Install dependencies**
   ```bash
   # Backend
   cd backend
   npm install

   # Frontend
   cd ../frontend
   npm install

   # Python API
   cd ../python-face-api
   pip install -r requirements.txt
   ```

3. **Configure environment**
   - Copy `backend/.env.example` to `backend/.env`
   - Copy `frontend/.env.local.example` to `frontend/.env.local`

4. **Start all services**
   
   **Windows:**
   ```bash
   START_SYSTEM.bat
   ```

   **Manual (3 terminals):**
   ```bash
   # Terminal 1: Backend
   cd backend
   node server.js

   # Terminal 2: Frontend
   cd frontend
   npm run dev

   # Terminal 3: Python API
   cd python-face-api
   python main.py
   ```

5. **Access the application**
   - Open browser: `http://localhost:3000`
   - Login with admin credentials:
     - Email: `admin@attendance.com`
     - Password: `admin123`

## 📖 Documentation

- **[Complete Startup Guide](SYSTEM_STARTUP_GUIDE.md)** - Detailed setup instructions
- **[System Ready Guide](COMPLETE_SYSTEM_READY.md)** - Features and verification
- **[UI Upgrade Documentation](UI_UPGRADE_COMPLETE_FINAL.md)** - Design system details

## 🏗️ Architecture

```
┌─────────────────┐
│   Frontend      │  Next.js 14 + TypeScript
│  (Port 3000)    │  Tailwind CSS
└────────┬────────┘
         │
         │ REST API
         ▼
┌─────────────────┐
│   Backend       │  Node.js + Express
│  (Port 5000)    │  JWT Authentication
└────────┬────────┘
         │
         ├─────────────────┐
         │                 │
         ▼                 ▼
┌─────────────────┐  ┌─────────────────┐
│   MongoDB       │  │  Python API     │
│  (Port 27017)   │  │  (Port 8000)    │
│                 │  │  FastAPI        │
│  - Users        │  │  - face_recog   │
│  - Attendance   │  │  - OpenCV       │
│  - Timetable    │  │  - dlib         │
└─────────────────┘  └─────────────────┘
```

## 🎯 Usage Workflow

### 1. Admin Setup (One-time)
1. Login as admin
2. Add faculty members with subjects
3. Create timetable entries
4. Register students
5. Train student faces (50 images in 5 seconds)

### 2. Faculty Daily Use
1. Login as faculty
2. Go to "Live Attendance"
3. Select subject, year, and batch
4. Click "Start Live Recognition"
5. System auto-marks recognized students
6. Click "Stop & Complete Attendance"

### 3. Export Reports
1. Go to "Export Attendance"
2. Select date range and filters
3. Download Excel report

## 🔧 Technology Stack

### Frontend
- Next.js 14
- TypeScript
- Tailwind CSS
- Axios
- React Toastify

### Backend
- Node.js
- Express.js
- MongoDB + Mongoose
- JWT Authentication
- bcryptjs

### Face Recognition
- Python FastAPI
- face_recognition library
- OpenCV
- dlib
- NumPy

## 📊 Face Recognition Details

- **Training**: 50 images captured in 5 seconds
- **Recognition**: Processes 1 frame per second
- **Threshold**: Distance ≤ 0.35 for auto-marking
- **Accuracy**: High accuracy with proper lighting
- **Detection**: Colored boxes indicate recognition status
  - 🟢 Green: Excellent match (≤ 0.35)
  - 🟡 Yellow: Good match (0.35-0.50)
  - 🔴 Red: Unknown (> 0.50)
  - 🔵 Blue: Already marked

## 🔐 Security Features

- JWT-based authentication
- Password hashing with bcrypt
- Role-based access control (Admin/Faculty)
- Protected API routes
- Secure session management

## 📱 Responsive Design

- Mobile-first approach
- Works on all screen sizes
- Touch-friendly interface
- Optimized for tablets and desktops

## 🧪 Testing

Test all backend routes:
```bash
cd backend
node test-routes.js
```

## 🐛 Troubleshooting

### Common Issues

**"Failed to load dashboard"**
- Ensure backend is running
- Check MongoDB connection
- Verify faculty has subjects assigned

**"No trained students found"**
- Register students first
- Train their faces in Admin portal
- Verify training completed successfully

**"Camera not starting"**
- Allow camera permissions
- Check camera is not in use
- Try different browser (Chrome recommended)

**"Python API connection failed"**
- Ensure Python API is running
- Check port 8000 is not blocked
- Verify face_recognition is installed

## 📝 Default Credentials

### Admin
- Email: `admin@attendance.com`
- Password: `admin123`

### Faculty (after creation)
- Email: `aj@attendance.com`
- Password: `aj123`

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👥 Support

For issues and questions:
1. Check the documentation files
2. Run the test script
3. Review console logs
4. Check MongoDB connection

## 🎉 Acknowledgments

- face_recognition library by Adam Geitgey
- Next.js team for the amazing framework
- MongoDB for the database
- FastAPI for the Python API framework

---

**Made with ❤️ for educational institutions**

🚀 **Ready to revolutionize attendance management!**
