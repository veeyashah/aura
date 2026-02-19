const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const dotenv = require('dotenv');

dotenv.config();

const app = express();

// Middleware
app.use(cors({
  origin: process.env.CORS_ORIGIN || 'http://localhost:3000',
  credentials: true,
}));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Routes
app.use('/api/auth', require('./routes/auth'));
app.use('/api/admin', require('./routes/admin'));
app.use('/api/faculty', require('./routes/faculty'));
app.use('/api/attendance', require('./routes/attendance'));
app.use('/api/timetable', require('./routes/timetable'));

// Root health check endpoint (no /api prefix)
app.get('/health', (req, res) => {
  res.json({ status: 'OK', message: 'Attendance System running', timestamp: new Date().toISOString() });
});

// Health check at /api/health (for backward compatibility)
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', message: 'Server is running', timestamp: new Date().toISOString() });
});

// Handle 404 for API routes
app.use('/api/*', (req, res) => {
  res.status(404).json({ message: 'API route not found', path: req.path });
});

// Root API info endpoint
app.get('/', (req, res) => {
  res.json({ 
    message: 'Attendance System API', 
    status: 'running', 
    version: '1.0.0',
    endpoints: {
      health: '/health',
      api: '/api',
      admin: '/api/admin',
      auth: '/api/auth',
      faculty: '/api/faculty',
      attendance: '/api/attendance',
      timetable: '/api/timetable'
    }
  });
});

// Catch-all for unmatched routes
app.use('*', (req, res) => {
  res.status(404).json({ message: 'Route not found', path: req.path });
});

// Connect to MongoDB
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/attendance_system', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
.then(() => {
  console.log('✅ Connected to MongoDB');
  
  // Initialize default admin and timetables
  require('./utils/initialize').initializeDefaults();
})
.catch((error) => {
  console.error('❌ MongoDB connection error:', error);
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
