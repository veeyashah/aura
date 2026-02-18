const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

const userSchema = new mongoose.Schema({
  email: {
    type: String,
    required: true,
    unique: true,
    lowercase: true,
    trim: true,
  },
  password: {
    type: String,
    required: true,
  },
  role: {
    type: String,
    enum: ['admin', 'faculty', 'student'],
    required: true,
  },
  // Student specific fields
  studentId: {
    type: String,
    unique: true,
    sparse: true,
  },
  rollNumber: {
    type: String,
    unique: true,
    sparse: true,
  },
  name: {
    type: String,
  },
  year: {
    type: String,
    enum: ['FY', 'SY', 'TY'],
  },
  batch: {
    type: String,
    enum: ['B1', 'B2'],
  },
  department: {
    type: String,
  },
  course: {
    type: String,
  },
  contact: {
    type: String,
  },
  // Faculty specific fields
  facultyName: {
    type: String,
  },
  subjects: [{
    type: String,
  }],
  // Training status
  isTrained: {
    type: Boolean,
    default: false,
  },
  lastTrainedDate: {
    type: Date,
    default: null,
  },
  enrollmentDate: {
    type: Date,
    default: null,
  },
  faceEmbeddings: {
    type: [Number],  // Flat array of 512 numbers (ArcFace embeddings)
    default: [],
  },
  // Signature
  signature: {
    type: String, // Base64 encoded JPEG
    default: null,
  },
  signatureSubmitted: {
    type: Boolean,
    default: false,
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
});

userSchema.pre('save', async function(next) {
  if (!this.isModified('password')) return next();
  this.password = await bcrypt.hash(this.password, 10);
  next();
});

userSchema.methods.comparePassword = async function(candidatePassword) {
  return bcrypt.compare(candidatePassword, this.password);
};

module.exports = mongoose.model('User', userSchema);
