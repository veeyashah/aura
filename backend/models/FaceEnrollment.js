const mongoose = require('mongoose');

const faceEnrollmentSchema = new mongoose.Schema({
  studentId: {
    type: String,
    required: true,
    unique: true,
    index: true
  },
  enrollmentDate: {
    type: Date,
    default: Date.now
  },
  faceEmbeddings: [Number],  // 128-d array from face_recognition
  faceImages: [{
    imageBase64: String,
    capturedAt: Date
  }],
  status: {
    type: String,
    enum: ['pending', 'enrolled', 'failed'],
    default: 'pending'
  },
  enrollmentAttempts: {
    type: Number,
    default: 0
  },
  lastEnrollmentAttempt: Date,
  enrolledBy: mongoose.Schema.Types.ObjectId,  // Admin who verified
  createdAt: {
    type: Date,
    default: Date.now
  },
  updatedAt: {
    type: Date,
    default: Date.now
  }
}, { timestamps: true });

module.exports = mongoose.model('FaceEnrollment', faceEnrollmentSchema);
