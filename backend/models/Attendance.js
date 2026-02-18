const mongoose = require('mongoose');

const attendanceSchema = new mongoose.Schema({
  studentId: {
    type: String,
    required: true,
  },
  studentName: {
    type: String,
    required: true,
  },
  facultyName: {
    type: String,
    required: true,
  },
  subject: {
    type: String,
    required: true,
  },
  lectureType: {
    type: String,
    enum: ['Lecture', 'Practical', 'Tutorial'],
    default: 'Lecture',
  },
  division: {
    type: String,
  },
  year: {
    type: String,
    enum: ['FY', 'SY', 'TY'],
    required: true,
  },
  batch: {
    type: String,
    enum: ['B1', 'B2', 'ALL'],
    required: false,
  },
  date: {
    type: Date,
    required: true,
  },
  timeIn: Date,
  timeOut: Date,
  status: {
    type: String,
    enum: ['P', 'A', 'AG'],
    required: true,
  },
  markedBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
  },
  confidence: {
    type: Number,
    default: null,
  },
  // CONFIDENCE THRESHOLD TRACKING
  confidenceThreshold: {
    type: String,
    enum: ['auto-marked', 'manual-confirm', 'manual-review'],
    default: null,
  },
  // STUDENT SIGNATURE INITIALS (denormalized from User for fast export)
  studentSignatureInitials: {
    type: String,
    default: '',
  },
  // STUDENT ROLL NUMBER (denormalized from User for fast export and sorting)
  studentRollNumber: {
    type: String,
    default: '',
  },
  // FACE CAPTURE DATA
  faceCapture: {
    imageBase64: String,
    timestamp: Date,
    recognized: Boolean,
    confidence: Number,
    location: {
      top: Number,
      right: Number,
      bottom: Number,
      left: Number
    }
  },
  notes: {
    type: String,
    default: ''
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
}, { timestamps: true });

// Add indexes for faster queries
attendanceSchema.index({ studentId: 1, date: 1, subject: 1 });
attendanceSchema.index({ facultyName: 1, date: 1 });
attendanceSchema.index({ date: 1 }); // For sorting
attendanceSchema.index({ 'faceCapture.timestamp': 1 }); // For face lookups
attendanceSchema.index({ studentId: 1 }); // For student filtering

module.exports = mongoose.model('Attendance', attendanceSchema);
