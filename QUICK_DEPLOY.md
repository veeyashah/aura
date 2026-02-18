# QUICK DEPLOYMENT GUIDE

## One-Time Setup (Only if not done yet)

### 1. Render Backend Environment Variables
Go to: https://dashboard.render.com → aura-backend → Environment

Click "Edit" and set these values:
```
CORS_ORIGIN = https://aura-eight-beta.vercel.app
JWT_SECRET = auraSuperSecretKey123
MONGODB_URI = mongodb+srv://meveeyashah_db_user:Veeyashah123@auracluster.wodc7rq.mongodb.net/?appName=auracluster
NODE_ENV = production
PORT = 5000
```

Save and the service auto-restarts.

### 2. Vercel Frontend Environment Variables
Go to: https://vercel.com → Select aura project → Settings → Environment Variables

Click "Add" and set these values:
```
NEXT_PUBLIC_BACKEND_URL = https://aura-backend-7tow.onrender.com
NEXT_PUBLIC_PYTHON_API_URL = https://aura-face-api.onrender.com
```

Save. You'll need to redeploy for these to take effect.

### 3. MongoDB Atlas IP Whitelist
Go to: https://cloud.mongodb.com → Network Access → IP Access List

Verify `0.0.0.0/0` is in the list (or specific IPs if restrictive).

---

## Deploy After Making Changes

### Step 1: Commit & Push Code
```bash
cd c:\AURA
git add .
git commit -m "A-Grade Facial Recognition System - Optimized Training & Recognition"
git push origin main
```

### Step 2: Render Auto-Deploys
- Render watches main branch
- Auto-deploys on push
- Takes 2-3 minutes
- Check: https://dashboard.render.com → aura-backend → Logs
- Look for: "✅ Connected to MongoDB"

### Step 3: Vercel Redeploy
Option A - Manual Redeploy:
- Go to: https://vercel.com → aura → Deployments
- Click latest deployment → "Redeploy" button
- Takes 1-2 minutes
- Check for green checkmark

Option B - Auto on Push:
- Vercel auto-deploys on push
- Takes 1-2 minutes
- Check: https://aura-eight-beta.vercel.app

### Step 4: Test
```
Browser:
1. Open https://aura-eight-beta.vercel.app
2. Open DevTools (F12) → Console tab
3. Login with admin credentials
4. Navigate to: Admin → Students
5. Click any student → Training
6. Training should complete in ~50 seconds
7. Check console for ✅ Success messages (no ❌ errors)
```

---

## Verify Deployment

### Health Checks
```bash
# 1. Backend health
curl https://aura-backend-7tow.onrender.com/api/health
# Expected: {"status":"OK","message":"Server is running"}

# 2. Database connection (via login)
# Try to login at: https://aura-eight-beta.vercel.app
# If login works, DB is connected

# 3. Frontend health
# Check: https://aura-eight-beta.vercel.app
# Should load without errors
```

### Browser Console Checks
After deploying, open the app and check console (F12):
```
✅ Expected:
- No error messages
- "🎓 Training student..." messages
- "✅ Training completed..." messages

❌ Problems if you see:
- "API route not found" → Check environment variables
- "Python API unreachable" → Check NEXT_PUBLIC_PYTHON_API_URL
- "TypeError: ..." → Check JavaScript syntax errors
```

---

## Troubleshooting Quick Fixes

### Problem: "API route not found"
**Fix**: 
1. Clear Vercel cache: Vercel Dashboard → Settings → Git → Disconnect/Reconnect
2. Wait 5 minutes and redeploy
3. Verify NEXT_PUBLIC_BACKEND_URL is set in Vercel

### Problem: "Python API unreachable"
**Fix**:
1. Verify NEXT_PUBLIC_PYTHON_API_URL matches exactly
2. Check Python API is running: curl https://aura-face-api.onrender.com/health
3. If 502 error, Render Python API might be down - restart it

### Problem: Training takes >3 minutes and times out
**Fix**:
1. Reduce camera movement during capture
2. Check internet speed (should be >1Mbps)
3. Check Render CPU usage: might need higher tier
4. Training should complete in 50 seconds if optimized correctly

### Problem: Can't login
**Fix**:
1. Check MongoDB connection: Render logs should show "✅ Connected to MongoDB"
2. Check IP whitelist in MongoDB Atlas
3. Verify MONGODB_URI env var spelling exactly
4. Check email/password aren't typos

---

## Rolling Back If Needed

### Revert Last Deploy
```bash
# See what changed
git log --oneline -5

# Revert to previous version
git revert <commit-hash>
git push origin main

# Both Render & Vercel auto-redeploy within 2-3 minutes
```

### Render Previous Version
- Dashboard → aura-backend → Deployments
- Click previous green deployment
- Click "..." → Choose specific version
- No restart needed, just click it

### Vercel Previous Version
- Dashboard → aura → Deployments
- Click previous deployment with ✅ checkmark
- Auto-promotes to Production

---

## Success Indicators ✅

After full deployment, you should see:

1. **Training Page** (Admin → Students → Select → Training)
   ```
   ✅ Camera starts within 2 seconds
   ✅ "Capturing... Keep face steady!" appears
   ✅ Shows "10 images captured"
   ✅ Modal shows "Training in Progress..."
   ✅ Completes within 50 seconds
   ✅ Shows "Training completed! 512 dimensions saved"
   ✅ Redirects to students list
   ```

2. **Live Attendance** (Faculty → Live Attendance)
   ```
   ✅ Subject dropdown loads
   ✅ "Start Live Recognition" button works
   ✅ Camera preview shows
   ✅ Faces detected within 1 second
   ✅ Auto-marks with confidence >0.65
   ✅ Manual review for confidence 0.4-0.65
   ```

3. **Console** (F12 → Console)
   ```
   ✅ No red error messages
   ✅ Blue info messages show progress
   ✅ Network tab shows 200/201 responses
   ```

---

## Performance Targets ✅

These should be met after optimization:

| Metric | Target | How to Check |
|--------|--------|------------|
| Training time | <60s | Run training, time from click to success message |
| Recognition latency | <400ms | Check Network tab → recognize endpoint response time |
| False positive rate | <1% | Test with wrong person's face 10 times |
| AI accuracy | >99% | Test with 10 different trained students |

---

## Emergency Support

If something breaks in production:

### Never Do
❌ Don't modify database directly
❌ Don't change model imports
❌ Don't modify API routes
❌ Don't change authentication flow

### Always Do
✅ Check Render logs first
✅ Check Vercel deployment logs
✅ Check browser console (F12)
✅ Check MongoDB Atlas for connection errors
✅ Revert to last known good commit if unsure

### Contact
- Render Status: https://status.render.com/
- Vercel Status: https://www.vercel-status.com/
- MongoDB Status: https://status.mongodb.com/

---

## Deployment Checklist

Before saying "it's done":

- [ ] Render environment variables all set
- [ ] Vercel environment variables all set
- [ ] MongoDB IP whitelist allows Render
- [ ] Backend logs show "✅ Connected to MongoDB"
- [ ] Frontend loads without errors
- [ ] Login works
- [ ] Training completes in <60 seconds
- [ ] Recognition works in live attendance
- [ ] Console has no error messages

---

**Last Updated**: Feb 18, 2026
**Status**: ✅ SYSTEM IS A-GRADE READY
