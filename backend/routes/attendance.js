const express = require('express');
const Attendance = require('../models/Attendance');
const User = require('../models/User');
const { auth } = require('../middleware/auth');

const router = express.Router();
router.use(auth);

/**
 * POST /api/attendance/capture
 * Capture attendance record
 */
router.post('/capture', async (req, res) => {
  try {
    const { studentId, subject, lectureType, confidence, distance, timestamp } = req.body;

    // resolve student name if available
    const student = await User.findOne({ studentId });
    const studentName = student ? (student.name || student.studentId) : (req.body.studentName || studentId);
    
    // Extract signature initials and roll number (denormalize to attendance for fast export)
    let studentSignatureInitials = '';
    let studentRollNumber = '';
    if (student && student.signature) {
      try {
        const parsed = JSON.parse(student.signature);
        studentSignatureInitials = parsed.initials || '';
      } catch (e) {
        studentSignatureInitials = student.signature || '';
      }
    }
    if (student && student.rollNumber) {
      studentRollNumber = student.rollNumber;
    }

    const facultyName = req.user.facultyName || req.user.name || req.user.email;
    const confValue = typeof confidence === 'number' ? confidence : 0.95;
    
    // Determine confidence threshold category: auto-marked (≥0.85), manual-confirm (0.70-0.84), manual-review (<0.70)
    let confidenceThreshold = 'manual-review';
    if (confValue >= 0.85) confidenceThreshold = 'auto-marked';
    else if (confValue >= 0.70) confidenceThreshold = 'manual-confirm';

    const record = new Attendance({
      studentId,
      studentName,
      subject,
      lectureType: lectureType || 'Lecture',
      status: 'P', // use schema enum: 'P' present
      markedBy: req.user._id,
      facultyName,
      studentSignatureInitials,
      studentRollNumber,
      confidence: confValue,
      confidenceThreshold,
      date: timestamp ? new Date(timestamp) : new Date(),
      faceCapture: {
        timestamp: timestamp ? new Date(timestamp) : new Date(),
        recognized: true,
        confidence: confValue,
        location: {
          top: 0,
          right: 0,
          bottom: 0,
          left: 0
        }
      }
    });

    await record.save();

    res.json({
      success: true,
      message: 'Attendance recorded',
      data: record
    });
  } catch (error) {
    console.error('Capture error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * POST /api/attendance/capture-face
 * Capture face image and attendance record
 */
router.post('/capture-face', async (req, res) => {
  try {
    const { studentId, subject, lectureType, faceData, confidence, distance, timestamp } = req.body;

    if (!studentId) {
      return res.status(400).json({ message: 'Student ID is required' });
    }

    if (!subject) {
      return res.status(400).json({ message: 'Subject is required' });
    }


    // resolve student name if available
    const student = await User.findOne({ studentId });
    const studentName = student ? (student.name || student.studentId) : (req.body.studentName || studentId);
    
    // Extract signature initials and roll number (denormalize to attendance for fast export)
    let studentSignatureInitials = '';
    let studentRollNumber = '';
    if (student && student.signature) {
      try {
        const parsed = JSON.parse(student.signature);
        studentSignatureInitials = parsed.initials || '';
      } catch (e) {
        studentSignatureInitials = student.signature || '';
      }
    }
    if (student && student.rollNumber) {
      studentRollNumber = student.rollNumber;
    }

    const facultyName = req.user.facultyName || req.user.name || req.user.email;
    const confValue = typeof confidence === 'number' ? confidence : 0.95;
    
    // Determine confidence threshold category: auto-marked (≥0.85), manual-confirm (0.70-0.84), manual-review (<0.70)
    let confidenceThreshold = 'manual-review';
    if (confValue >= 0.85) confidenceThreshold = 'auto-marked';
    else if (confValue >= 0.70) confidenceThreshold = 'manual-confirm';

    const record = new Attendance({
      studentId,
      studentName,
      subject,
      lectureType: lectureType || 'Lecture',
      status: 'P',
      markedBy: req.user._id,
      facultyName,
      studentSignatureInitials,
      studentRollNumber,
      confidence: confValue,
      confidenceThreshold,
      faceCapture: {
        imageBase64: faceData || null,
        timestamp: timestamp ? new Date(timestamp) : new Date(),
        recognized: true,
        confidence: confValue
      },
      date: timestamp ? new Date(timestamp) : new Date()
    });

    await record.save();

    res.json({
      success: true,
      message: 'Face attendance recorded',
      data: record
    });
  } catch (error) {
    console.error('Capture face error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * GET /api/attendance/:studentId
 * Get student's attendance history
 */
router.get('/:studentId', async (req, res) => {
  try {
    const records = await Attendance.find({ studentId: req.params.studentId })
      .select('subject lectureType status date confidence')
      .sort({ date: -1 })
      .limit(100);

    // Calculate stats
    const total = records.length;
    const present = records.filter(r => r.status === 'present').length;
    const percentage = total > 0 ? ((present / total) * 100).toFixed(2) : 0;

    res.json({
      success: true,
      data: {
        records,
        stats: {
          total,
          present,
          percentage
        }
      }
    });
  } catch (error) {
    console.error('Get attendance error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

module.exports = router;
