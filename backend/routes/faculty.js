const express = require('express');
const XLSX = require('xlsx');
const ExcelJS = require('exceljs');
const User = require('../models/User');
const Attendance = require('../models/Attendance');
const Timetable = require('../models/Timetable');
const { auth, authorize } = require('../middleware/auth');

const router = express.Router();

// All faculty routes require authentication and faculty role
router.use(auth);


/**
 * GET /api/faculty/students
 * Get trained students for faculty (filtered by assigned subjects)
 */
// All remaining faculty routes require faculty role
router.use(authorize('faculty'))

router.get('/students', async (req, res) => {
  try {
    const { year, batch } = req.query;

    let query = { role: 'student', isTrained: true };
    if (year) query.year = year;
    if (batch && batch !== 'ALL') query.batch = batch;

    const students = await User.find(query)
      .select('studentId name rollNumber year batch faceEmbeddings isTrained')
      .lean();

    // Format for frontend - send embeddings as array
    const formatted = students.map(s => ({
      studentId: s.studentId,
      name: s.name,
      rollNumber: s.rollNumber,
      year: s.year,
      batch: s.batch,
      isTrained: s.isTrained,
      faceEmbeddings: s.faceEmbeddings || []
    }));

    res.json(formatted);
  } catch (error) {
    console.error('Get students error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * GET /api/faculty/timetable
 * Get timetable for faculty (all their classes)
 */
router.get('/timetable', async (req, res) => {
  try {
    const facultyName = req.user.facultyName || req.user.name;
    
    const timetable = await Timetable.find({
      facultyName: facultyName
    }).sort({ day: 1, timeSlot: 1 });

    res.json(timetable);
  } catch (error) {
    console.error('Get timetable error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * GET /api/faculty/subjects
 * Get subjects assigned to faculty
 */
router.get('/subjects', async (req, res) => {
  try {
    const facultyName = req.user.facultyName || req.user.name;
    
    const subjects = await Timetable.distinct('subject', {
      facultyName: facultyName
    });

    res.json(subjects);
  } catch (error) {
    console.error('Get subjects error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * POST /api/faculty/attendance/submit
 * Submit attendance for multiple students
 */
router.post('/attendance/submit', async (req, res) => {
  try {
    const { subject, lectureType, studentAttendance, date } = req.body;

    if (!subject || !lectureType || !studentAttendance || studentAttendance.length === 0) {
      return res.status(400).json({ message: 'Missing required fields' });
    }

    // Save attendance records
    const facultyName = req.user.facultyName || req.user.name || req.user.email;
    const records = studentAttendance.map(item => ({
      studentId: item.studentId,
      studentName: item.studentName || item.name || item.studentId,
      subject,
      lectureType,
      status: 'P',
      markedBy: req.user._id,
      facultyName,
      date: date || new Date(),
      confidence: item.confidence || 0.95,
      faceDistance: item.distance || 0.3
    }));

    await Attendance.insertMany(records);

    res.json({
      success: true,
      message: `Attendance submitted for ${records.length} students`,
      count: records.length
    });
  } catch (error) {
    console.error('Submit attendance error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * GET /api/faculty/attendance
 * Get attendance history
 */
router.get('/attendance', async (req, res) => {
  try {
    const { subject, startDate, endDate } = req.query;

    let query = { markedBy: req.user._id };
    if (subject) query.subject = subject;
    if (startDate || endDate) {
      query.date = {};
      if (startDate) query.date.$gte = new Date(startDate);
      if (endDate) query.date.$lte = new Date(endDate);
    }

    const records = await Attendance.find(query)
      .select('studentId subject lectureType status date confidence')
      .sort({ date: -1 })
      .limit(1000);

    res.json(records);
  } catch (error) {
    console.error('Get attendance error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * POST /api/faculty/attendance
 * Mark attendance for a single student (faculty and admin can mark)
 */
router.post('/attendance', async (req, res) => {
  // Allow both faculty and admin roles
  if (!['faculty', 'admin'].includes(req.user.role)) {
    return res.status(403).json({ message: 'Access denied' })
  }
  try {
    const { studentId, studentName, subject, status, lectureType, year, batch, division, confidence, confidenceThreshold } = req.body;

    if (!studentId || !subject || !status) {
      return res.status(400).json({ message: 'Missing required fields' });
    }

    // Check if already marked today
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    const existing = await Attendance.findOne({
      studentId,
      subject,
      lectureType: lectureType || 'Lecture',
      date: { $gte: today, $lt: tomorrow }
    });

    if (existing) {
      return res.json({
        success: true,
        alreadyMarked: true,
        message: 'Attendance already marked for this student today'
      });
    }

    // Create new attendance record
    const facultyName = req.user.facultyName || req.user.name || req.user.email;

    const record = new Attendance({
      studentId,
      studentName,
      subject,
      status,
      lectureType: lectureType || 'Lecture',
      year,
      batch,
      division,
      markedBy: req.user._id,
      facultyName,
      confidence: confidence || 1.0,
      confidenceThreshold: confidenceThreshold || 'manual-review',
      date: new Date()
    });

    await record.save();

    res.json({
      success: true,
      message: 'Attendance marked successfully',
      data: record
    });
  } catch (error) {
    console.error('Mark attendance error:', error);
    res.status(500).json({ message: error.message || 'Server error' });
  }
});

/**
 * GET /api/faculty/dashboard
 * Faculty dashboard stats
 */
router.get('/dashboard', async (req, res) => {
  try {
    console.log('=== Faculty Dashboard Request ===');
    console.log('User:', req.user.email);
    console.log('Faculty Name:', req.user.facultyName);
    console.log('Name:', req.user.name);
    
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    // Count today's attendance records marked by this faculty
    let todayActivity = 0;
    try {
      todayActivity = await Attendance.countDocuments({
        markedBy: req.user._id,
        createdAt: { $gte: today }
      });
      console.log('Today Activity:', todayActivity);
    } catch (err) {
      console.error('Error counting today activity:', err);
    }

    // Get subject-wise attendance statistics
    const facultyName = req.user.facultyName || req.user.name;
    console.log('Looking for timetable with facultyName:', facultyName);
    
    let subjects = [];
    try {
      subjects = await Timetable.distinct('subject', { facultyName: facultyName });
      console.log('Found subjects:', subjects);
    } catch (err) {
      console.error('Error getting subjects:', err);
    }
    
    const stats = [];
    for (const subject of subjects) {
      try {
        const totalRecords = await Attendance.countDocuments({
          subject: subject,
          markedBy: req.user._id
        });
        
        const presentRecords = await Attendance.countDocuments({
          subject: subject,
          markedBy: req.user._id,
          status: 'P'
        });
        
        const percentage = totalRecords > 0 ? ((presentRecords / totalRecords) * 100).toFixed(1) : '0.0';
        
        stats.push({
          subject: subject,
          total: totalRecords,
          present: presentRecords,
          percentage: percentage
        });
      } catch (err) {
        console.error(`Error processing subject ${subject}:`, err);
      }
    }

    const response = {
      todayActivity: todayActivity,
      stats: stats,
      facultyName: facultyName
    };
    
    console.log('Sending response:', JSON.stringify(response));
    console.log('=== End Dashboard Request ===\n');
    
    res.json(response);
  } catch (error) {
    console.error('=== Dashboard Error ===');
    console.error('Error:', error);
    console.error('Stack:', error.stack);
    res.status(500).json({ 
      message: error.message || 'Server error',
      error: process.env.NODE_ENV === 'development' ? error.stack : undefined
    });
  }
});

/**
 * GET /api/faculty/export-attendance
 * Export attendance records marked by this faculty as Excel file (with signatures in Vladimir Script)
 */
router.get('/export-attendance', async (req, res) => {
  try {
    const { subject, startDate, endDate } = req.query;
    
    let query = { markedBy: req.user._id };
    if (subject) query.subject = subject;
    if (startDate || endDate) {
      query.date = {};
      if (startDate) query.date.$gte = new Date(startDate);
      if (endDate) query.date.$lte = new Date(endDate);
    }
    
    // Use aggregation to JOIN with User collection to get roll number and signature
    const records = await Attendance.aggregate([
      { $match: query },
      {
        $lookup: {
          from: 'users',
          localField: 'studentId',
          foreignField: 'studentId',
          as: 'student'
        }
      },
      {
        $unwind: {
          path: '$student',
          preserveNullAndEmptyArrays: true
        }
      },
      {
        $project: {
          studentId: 1,
          studentName: 1,
          // Prefer User's roll number, fallback to denormalized
          studentRollNumber: { $ifNull: ['$student.rollNumber', '$studentRollNumber'] },
          // Use denormalized signature initials from Attendance record
          studentSignatureInitials: '$studentSignatureInitials',
          subject: 1,
          lectureType: 1,
          status: 1,
          date: 1
        }
      },
      { $sort: { date: -1, _id: -1 } }
    ]);
    
    if (records.length === 0) {
      return res.status(404).json({ message: 'No attendance records found' });
    }
    
    // Generate signature initials from student name if not set
    records.forEach(record => {
      if (!record.studentSignatureInitials || record.studentSignatureInitials.trim() === '') {
        // Extract initials from name: "Atharva Panindre" -> "AP"
        const nameParts = (record.studentName || '').split(' ').filter(part => part.length > 0);
        if (nameParts.length >= 2) {
          record.studentSignatureInitials = (nameParts[0][0] + nameParts[1][0]).toUpperCase();
        } else if (nameParts.length === 1) {
          record.studentSignatureInitials = nameParts[0][0].toUpperCase();
        } else {
          record.studentSignatureInitials = '';
        }
      }
    });
    
    // Create Excel workbook with ExcelJS
    const workbook = new ExcelJS.Workbook();
    const worksheet = workbook.addWorksheet('Attendance');

    // Define header row with safe column definitions
    const headers = ['Roll Number', 'Student ID', 'Student Name', 'Subject', 'Lecture Type', 'Status', 'Date', 'Signature'];
    
    // Set column widths
    worksheet.columns = [
      { header: 'Roll Number', width: 14 },
      { header: 'Student ID', width: 14 },
      { header: 'Student Name', width: 22 },
      { header: 'Subject', width: 22 },
      { header: 'Lecture Type', width: 15 },
      { header: 'Status', width: 10 },
      { header: 'Date', width: 14 },
      { header: 'Signature', width: 25 }
    ];
    
    const headerRow = worksheet.getRow(1);
    
    // Style header row
    headerRow.eachCell((cell, colNumber) => {
      cell.font = { name: 'Arial', size: 11, bold: true, color: { argb: 'FFFFFFFF' } };
      cell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF366092' } };
      cell.alignment = { horizontal: 'center', vertical: 'center', wrapText: true };
      cell.border = {
        top: { style: 'thin', color: { argb: 'FF000000' } },
        left: { style: 'thin', color: { argb: 'FF000000' } },
        bottom: { style: 'thin', color: { argb: 'FF000000' } },
        right: { style: 'thin', color: { argb: 'FF000000' } }
      };
    });

    // Add data rows
    records.forEach(record => {
      const rowData = [
        record.studentRollNumber || '',
        record.studentId || '',
        record.studentName || '',
        record.subject || '',
        record.lectureType || '',
        record.status === 'P' ? 'Present' : record.status === 'A' ? 'Absent' : (record.status || ''),
        record.date ? record.date.toISOString().split('T')[0] : '',
        record.studentSignatureInitials || ''
      ];

      const row = worksheet.addRow(rowData);

      // Style data rows
      row.eachCell((cell, colNumber) => {
        // Signature column - use Vladimir Script for elegant handwritten look
        if (colNumber === 8) {
          cell.font = { name: 'Vladimir Script', size: 14, bold: true, color: { argb: 'FF000000' } };
        } else {
          cell.font = { name: 'Calibri', size: 11 };
        }

        cell.alignment = { 
          horizontal: colNumber === 8 ? 'center' : 'left', 
          vertical: 'center',
          wrapText: false
        };
        
        // Light borders for data rows
        cell.border = {
          top: { style: 'thin', color: { argb: 'FFCCCCCC' } },
          left: { style: 'thin', color: { argb: 'FFCCCCCC' } },
          bottom: { style: 'thin', color: { argb: 'FFCCCCCC' } },
          right: { style: 'thin', color: { argb: 'FFCCCCCC' } }
        };
      });
    });

    // Freeze first row
    worksheet.views = [{ state: 'frozen', ySplit: 1 }];

    // Generate Excel buffer
    const buffer = await workbook.xlsx.writeBuffer();
    
    res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
    res.setHeader('Content-Disposition', `attachment; filename="attendance_${new Date().toISOString().split('T')[0]}.xlsx"`);
    res.send(buffer);
  } catch (error) {
    console.error('Export attendance error:', error);
    res.status(500).json({ message: 'Export failed' });
  }
});

module.exports = router;
