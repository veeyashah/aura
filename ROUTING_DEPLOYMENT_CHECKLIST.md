# Routing Deployment Checklist

After the backend/frontend routing refactor, use this checklist to verify everything works correctly.

## Pre-Deployment Testing (Local)

### 1. Backend Health Checks
```bash
# Terminal 1: Start backend
cd backend
npm start

# Terminal 2: Test health endpoints
curl http://localhost:5000/health
curl http://localhost:5000/api/health
curl http://localhost:5000/
```

**Expected Results:**
- `GET /health` → `{ status: 'OK', message: 'Attendance System running', ... }`
- `GET /api/health` → `{ status: 'OK', message: 'Server is running', ... }`
- `GET /` → Shows API info with all endpoints listed

### 2. Frontend API Configuration (Local)
```bash
# Terminal 3: Start frontend (should use http://localhost:5000 as default backend)
cd frontend
npm run dev
```

**Check in Browser DevTools (Network tab):**
1. Navigate to Admin Dashboard
2. Look for API requests
3. Verify requests go to `http://localhost:5000/api/admin/dashboard` (NOT `http://localhost:5000/api/api/admin/dashboard`)

### 3. Sample Route Tests (Local)
```bash
# Test admin routes
curl -H "Authorization: Bearer <YOUR_JWT_TOKEN>" http://localhost:5000/api/admin/dashboard

# Test faculty routes
curl -H "Authorization: Bearer <YOUR_JWT_TOKEN>" http://localhost:5000/api/faculty/subjects

# Test invalid route (should 404)
curl http://localhost:5000/api/invalid/route
```

**Expected Results:**
- Valid routes return correct data
- Invalid routes return `{ message: 'API route not found', path: '/api/invalid/route' }`

## Production Deployment (Render + Vercel)

### Backend Environment Variables (Render)
```
MONGODB_URI=<your_mongodb_uri>
CORS_ORIGIN=https://your-frontend.vercel.app
JWT_SECRET=<your_jwt_secret>
NODE_ENV=production
PORT=5000  # Or leave empty; Render assigns automatically
```

### Frontend Environment Variables (Vercel)
```
NEXT_PUBLIC_BACKEND_URL=https://your-backend.onrender.com
NEXT_PUBLIC_FACE_API_URL=https://your-python-api.onrender.com
```

### Production Testing Checklist

✅ **Step 1: Verify Backend Deployment**
- [ ] Access `https://your-backend.onrender.com/health` → Returns status OK
- [ ] Access `https://your-backend.onrender.com/api/health` → Returns status OK  
- [ ] Access `https://your-backend.onrender.com/` → Shows API endpoints

✅ **Step 2: Verify Frontend Deployment**
- [ ] Frontend deployed on Vercel with environment variables set
- [ ] Frontend loads successfully (not showing 404 or CORS errors)

✅ **Step 3: Check DevTools Network Tab (Production)**
1. Open `https://your-frontend.vercel.app` in Chrome
2. Open DevTools → Network tab
3. Navigate to Admin Dashboard
4. Check requests:
   - **DO SEE:** Requests to `https://your-backend.onrender.com/api/admin/dashboard`
   - **DO NOT SEE:** Requests to `https://your-backend.onrender.com/api/api/admin/dashboard` (double prefix bug)

✅ **Step 4: Test Login Flow**
- [ ] Login page loads
- [ ] POST to `/api/auth/login` succeeds with correct credentials
- [ ] JWT token stored in localStorage
- [ ] Redirect to dashboard works
- [ ] Dashboard loads student data via `/api/admin/students`

✅ **Step 5: Test Admin Functions**
- [ ] Admin Dashboard loads stats
- [ ] Student list loads via `/api/admin/students`
- [ ] Can register new student via `/api/admin/students`
- [ ] Can delete student via `/api/admin/students/:id`
- [ ] Can trigger training via `/api/admin/students/:id/training`

✅ **Step 6: Test Faculty Functions**
- [ ] Faculty Dashboard loads
- [ ] Can view live attendance via `/api/faculty/subjects` and `/api/faculty/students`
- [ ] Can export attendance via `/api/faculty/export-attendance`
- [ ] Face recognition works (calls Python API via `NEXT_PUBLIC_FACE_API_URL`)

✅ **Step 7: Backend Logs (Render)**
1. Go to Render dashboard
2. Open your backend service
3. Check recent logs should show:
   - `✅ Connected to MongoDB`
   - `🚀 Server running on port 5000` (or assigned port)
   - No 404 errors for valid routes
   - No double-prefix requests like `/api/api/*`

## Troubleshooting

### 404 Errors on Valid Routes
**Problem:** Requests to `/api/admin/dashboard` return 404
- **Cause 1:** Backend routes not mounted under `/api`
- **Cause 2:** Route file defines `/api/admin/dashboard` instead of just `/dashboard`
- **Fix:** Check [backend/server.js](backend/server.js) lines 19-23 for correct mounting

### Double /api Prefix Bug (Production Only)
**Problem:** Production requests go to `/api/api/admin/*` instead of `/api/admin/*`
- **Cause:** `NEXT_PUBLIC_BACKEND_URL` already ends with `/api` before configuration appends another
- **Fix:** Verify [frontend/lib/api.ts](frontend/lib/api.ts#L26-L27) uses:
  ```typescript
  const BACKEND_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:5000'
  const API_URL = `${BACKEND_BASE}/api`
  ```

### CORS Errors
**Problem:** Browser shows CORS error when calling backend
- **Cause 1:** `CORS_ORIGIN` env var doesn't match frontend URL
- **Cause 2:** Wrong backend URL in frontend .env
- **Fix:** Update `CORS_ORIGIN` on Render backend to match Vercel frontend URL

### Face API Connection Issues  
**Problem:** Training or live attendance fails (face detection)
- **Cause:** `NEXT_PUBLIC_FACE_API_URL` incorrect or Python API not running
- **Fix:** Verify Python API is deployed and `NEXT_PUBLIC_FACE_API_URL` points to it

## API Route Reference

| Method | Route | Auth | Purpose |
|--------|-------|------|---------|
| POST | `/api/auth/login` | No | User login |
| POST | `/api/auth/logout` | Yes | User logout |
| GET | `/api/admin/dashboard` | Admin | Dashboard stats |
| GET | `/api/admin/students` | Admin | List all students |
| POST | `/api/admin/students` | Admin | Register new student |
| POST | `/api/admin/students/:id/training` | Admin | Start training |
| GET | `/api/faculty/dashboard` | Faculty | Faculty dashboard |
| GET | `/api/faculty/subjects` | Faculty | Get assigned subjects |
| GET | `/api/faculty/students` | Faculty | Get students for subject |
| POST | `/api/faculty/attendance` | Faculty | Mark attendance |
| GET | `/api/attendance/records` | Faculty | Get attendance records |
| GET | `/api/timetable` | Any | Get timetable |

## Performance Notes

- Backend cold start on Render: ~5-10 seconds (first request slower)
- Python API cold start: ~12-20 seconds (Facenet512 model preload)
- Subsequent requests: <200ms for backend, <400ms for face recognition
- Frontend bundle includes TensorFlow.js (optional to remove) - 2-3MB uncompressed

## Success Indicators

✅ All API requests go to correct endpoints
✅ No double-prefix `/api/api/*` requests
✅ LOGIN works, JWT tokens stored
✅ DASHBOARD loads data correctly
✅ ATTENDANCE marking works
✅ TRAINING triggers face capture
✅ LIVE ATTENDANCE shows real-time faces
✅ EXPORT generates Excel files
✅ No 404 errors in Render logs for valid routes
