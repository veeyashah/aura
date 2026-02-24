const express = require('express');
const bcrypt = require('bcryptjs');
const User = require('../models/User');
const Attendance = require('../models/Attendance');
const Timetable = require('../models/Timetable');
const FaceEnrollment = require('../models/FaceEnrollment');
const { auth, authorize } = require('../middleware/auth');

const router = express.Router();

router.use(auth);
router.use(authorize('admin'));

/**
 * GET /api/admin/dashboard
 * Admin dashboard statistics
 */
router.get('/dashboard', async (req, res) => {
  try {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    // Count total students
    const totalStudents = await User.countDocuments({ role: 'student' });
    
    // Count trained students
    const trainedStudents = await User.countDocuments({ role: 'student', isTrained: true });
    
    // Count total faculty
    const totalFaculty = await User.countDocuments({ role: 'faculty' });
    
    // Count today's attendance
    const todayAttendance = await Attendance.countDocuments({
      date: { $gte: today, $lt: tomorrow }
    });
    
    // Count lecture attendance (all time)
    const lectureAttendance = await Attendance.countDocuments({
      lectureType: { $in: ['Lecture', 'Practical', 'Tutorial'] }
    });
    
    // Count event attendance (all time) - will be 0 since events removed
    const eventAttendance = 0;
    
    // Get recent attendance records (last 10) - ONLY FOR EXISTING STUDENTS
    const recentAttendance = await Attendance.aggregate([
      { $sort: { date: -1, createdAt: -1 } },
      {
        $lookup: {
          from: 'users',
          localField: 'studentId',
          foreignField: 'studentId',
          as: 'studentData'
        }
      },
      {
        $match: {
          // Only include records where student STILL EXISTS in database
          $expr: { $gt: [{ $size: '$studentData' }, 0] }
        }
      },
      { $limit: 10 },
      {
        $addFields: {
          // Use the student's current name from User collection
          studentName: { $arrayElemAt: ['$studentData.name', 0] }
        }
      },
      {
        $project: {
          studentName: 1,
          subject: 1,
          facultyName: 1,
          date: 1,
          status: 1,
          lectureType: 1
        }
      }
    ]);
    
    // Format recent attendance
    const formattedRecent = recentAttendance.map(record => ({
      studentName: record.studentName || 'Unknown',
      subject: record.subject || 'N/A',
      facultyName: record.facultyName || 'System',
      date: record.date,
      status: record.status || 'P',
      isEvent: false // No events anymore
    }));

    res.json({
      totalStudents,
      trainedStudents,
      totalFaculty,
      todayAttendance,
      lectureAttendance,
      eventAttendance,
      recentAttendance: formattedRecent
    });
  } catch (error) {
    console.error('Admin dashboard error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * GET /api/admin/students
 */
router.get('/students', async (req, res) => {
  try {
    const { page = 1, limit = 50, year, batch, trained } = req.query;
    const skip = (page - 1) * limit;

    let query = { role: 'student' };
    if (year) query.year = year;
    if (batch && batch !== 'ALL') query.batch = batch;
    if (trained === 'true') query.isTrained = true;
    if (trained === 'false') query.isTrained = false;

    const students = await User.find(query)
      .select('studentId rollNumber name year batch department isTrained faceEmbeddings enrollmentDate signature')
      .skip(skip)
      .limit(parseInt(limit))
      .sort({ name: 1 });

    const total = await User.countDocuments(query);

    res.json({
      success: true,
      data: students.map(s => ({
        _id: s._id,
        studentId: s.studentId,
        rollNumber: s.rollNumber || null,
        name: s.name,
        year: s.year,
        batch: s.batch,
        department: s.department || null,
        isTrained: s.isTrained,
        signature: s.signature || null,
        signatureSubmitted: !!s.signature,
        embeddingSize: s.faceEmbeddings ? s.faceEmbeddings.length : 0,
        enrollmentDate: s.enrollmentDate
      })),
      total,
      page: parseInt(page),
      pages: Math.ceil(total / limit)
    });

  } catch (error) {
    console.error('Get students error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * GET /api/admin/students/:id
 */
router.get('/students/:id', async (req, res) => {
  try {
    const student = await User.findById(req.params.id).select('-password');
    if (!student) {
      return res.status(404).json({ message: 'Student not found' });
    }

    const enrollment = await FaceEnrollment.findOne({ studentId: student.studentId });

    res.json({
      success: true,
      data: {
        ...student.toObject(),
        faceEmbeddings: student.faceEmbeddings ? {
          count: student.faceEmbeddings.length,
          lastTrained: student.lastTrainedDate
        } : null,
        enrollmentStatus: enrollment?.status || 'not_enrolled'
      }
    });

  } catch (error) {
    console.error('Get student error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * POST /api/admin/students/:id/training
 * Save training embeddings - OPTIMIZED for speed
 */
router.post('/students/:id/training', async (req, res) => {
  try {
    const { faceEmbeddings } = req.body;

    if (!faceEmbeddings || !Array.isArray(faceEmbeddings) || faceEmbeddings.length === 0) {
      return res.status(400).json({ message: 'Invalid embeddings' });
    }

    // Get user first to get studentId
    const user = await User.findById(req.params.id);
    if (!user) {
      return res.status(404).json({ message: 'Student not found' });
    }

    // Parallel update: User embeddings + FaceEnrollment (both fast)
    await Promise.all([
      // Update User with embeddings
      User.updateOne(
        { _id: req.params.id },
        {
          $set: {
            faceEmbeddings,
            isTrained: true,
            lastTrainedDate: new Date()
          }
        }
      ),
      // Update FaceEnrollment with embeddings
      FaceEnrollment.updateOne(
        { studentId: user.studentId },
        {
          $set: {
            studentId: user.studentId,
            faceEmbeddings,
            status: 'enrolled',
            enrollmentDate: new Date()
          }
        },
        { upsert: true }
      )
    ]);

    res.json({
      success: true,
      message: 'Student training completed successfully',
      data: {
        studentId: user.studentId,
        name: user.name,
        isTrained: true,
        embeddingSize: faceEmbeddings.length
      }
    });

  } catch (error) {
    console.error('Training save error:', error);
    const errorMessage = error.message || 'Failed to save training data';
    res.status(500).json({ 
      message: errorMessage,
      success: false
    });
  }
});

/**
 * POST /api/admin/students
 * Register new student (password auto-generated since students don't login)
 */
router.post('/students', async (req, res) => {
  try {
    const { studentId, rollNumber, name, year, batch, department, course, email, contact } = req.body;

    // Validation
    if (!studentId || !name || !year || !batch || !email) {
      return res.status(400).json({ message: 'Missing required fields' });
    }

    // Check if student already exists
    const existing = await User.findOne({ $or: [{ email }, { studentId }] });
    if (existing) {
      return res.status(409).json({ message: 'Student already exists with this email or ID' });
    }

    // Auto-generate password (students don't need to login, but field is required in schema)
    // Password will be hashed by the User model's pre-save hook
    const autoPassword = `student_${studentId}_${Date.now()}`;

    // Create student
    const student = new User({
      studentId,
      rollNumber,
      name,
      year,
      batch,
      department,
      course,
      email,
      contact,
      password: autoPassword,  // Will be hashed by pre-save hook
      role: 'student',
      isTrained: false,
      enrollmentDate: new Date()
    });

    await student.save();

    res.status(201).json({
      success: true,
      message: 'Student registered successfully',
      student: {
        _id: student._id,
        studentId: student.studentId,
        name: student.name,
        email: student.email,
        year: student.year,
        batch: student.batch
      }
    });

  } catch (error) {
    console.error('Student registration error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * DELETE /api/admin/students/:id
 * Delete a student
 */
router.delete('/students/:id', async (req, res) => {
  try {
    const student = await User.findByIdAndDelete(req.params.id);
    
    if (!student) {
      return res.status(404).json({ message: 'Student not found' });
    }

    // Also delete face enrollment records
    await FaceEnrollment.deleteMany({ studentId: student.studentId });
    
    // Delete attendance records
    await Attendance.deleteMany({ studentId: student.studentId });

    res.json({
      success: true,
      message: 'Student deleted successfully',
      data: {
        studentId: student.studentId,
        name: student.name
      }
    });
  } catch (error) {
    console.error('Delete student error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * PUT /api/admin/students/:id
 * Update student information (name, signature, etc.)
 */
router.put('/students/:id', async (req, res) => {
  try {
    const { name, signature, rollNumber, contact, department } = req.body;
    
    const updateData = {};
    if (name !== undefined) updateData.name = name;
    if (signature !== undefined) updateData.signature = signature;
    if (rollNumber !== undefined) updateData.rollNumber = rollNumber;
    if (contact !== undefined) updateData.contact = contact;
    if (department !== undefined) updateData.department = department;
    
    const student = await User.findByIdAndUpdate(
      req.params.id,
      updateData,
      { new: true }
    ).select('-password');
    
    if (!student) {
      return res.status(404).json({ message: 'Student not found' });
    }
    
    res.json({
      success: true,
      message: 'Student updated successfully',
      data: student
    });
  } catch (error) {
    console.error('Update student error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * POST /api/admin/students/:id/signature
 * Compatibility route: save student signature (frontend posts here)
 */
router.post('/students/:id/signature', async (req, res) => {
  try {
    const { signature } = req.body;

    if (signature === undefined || signature === null) {
      return res.status(400).json({ message: 'Missing signature' });
    }

    const student = await User.findByIdAndUpdate(
      req.params.id,
      { signature, signatureSubmitted: true },
      { new: true }
    ).select('-password');

    if (!student) {
      return res.status(404).json({ message: 'Student not found' });
    }

    res.json({ success: true, message: 'Signature saved', data: student });
  } catch (error) {
    console.error('Save signature error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * PUT /api/admin/students/:id/signature
 * Compatibility: frontend sends PUT to this path — accept and save signature
 */
router.put('/students/:id/signature', async (req, res) => {
  try {
    const { signature } = req.body;

    if (signature === undefined || signature === null) {
      return res.status(400).json({ message: 'Missing signature' });
    }

    const student = await User.findByIdAndUpdate(
      req.params.id,
      { signature, signatureSubmitted: true },
      { new: true }
    ).select('-password');

    if (!student) {
      return res.status(404).json({ message: 'Student not found' });
    }

    res.json({ success: true, message: 'Signature saved', data: student });
  } catch (error) {
    console.error('Save signature (PUT) error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * GET /api/admin/timetable
 * Get timetable entries (with optional year filter)
 */
router.get('/timetable', async (req, res) => {
  try {
    const { year } = req.query;
    
    let query = {};
    if (year) query.year = year;
    
    const timetable = await Timetable.find(query)
      .sort({ day: 1, timeSlot: 1 });
    
    res.json(timetable);
  } catch (error) {
    console.error('Get timetable error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * POST /api/admin/timetable
 * Create new timetable entry
 */
router.post('/timetable', async (req, res) => {
  try {
    const { year, day, timeSlot, facultyName, subject, lectureType, division, batch } = req.body;
    
    if (!year || !day || !timeSlot || !facultyName || !subject || !batch) {
      return res.status(400).json({ message: 'Missing required fields' });
    }
    
    const entry = new Timetable({
      year,
      day,
      timeSlot,
      facultyName,
      subject,
      lectureType: lectureType || 'Lecture',
      division: division || 'A',
      batch
    });
    
    await entry.save();
    
    res.json({
      success: true,
      message: 'Timetable entry created',
      data: entry
    });
  } catch (error) {
    console.error('Create timetable error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * PUT /api/admin/timetable/:id
 * Update timetable entry
 */
router.put('/timetable/:id', async (req, res) => {
  try {
    const { year, day, timeSlot, facultyName, subject, lectureType, division, batch } = req.body;
    
    const entry = await Timetable.findByIdAndUpdate(
      req.params.id,
      { year, day, timeSlot, facultyName, subject, lectureType, division, batch },
      { new: true }
    );
    
    if (!entry) {
      return res.status(404).json({ message: 'Timetable entry not found' });
    }
    
    res.json({
      success: true,
      message: 'Timetable entry updated',
      data: entry
    });
  } catch (error) {
    console.error('Update timetable error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * DELETE /api/admin/timetable/:id
 * Delete timetable entry
 */
router.delete('/timetable/:id', async (req, res) => {
  try {
    const entry = await Timetable.findByIdAndDelete(req.params.id);
    
    if (!entry) {
      return res.status(404).json({ message: 'Timetable entry not found' });
    }
    
    res.json({
      success: true,
      message: 'Timetable entry deleted'
    });
  } catch (error) {
    console.error('Delete timetable error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * GET /api/admin/faculty
 * Get all faculty members with subjects from Timetable
 */
router.get('/faculty', async (req, res) => {
  try {
    // Use aggregation to fetch subjects from Timetable collection
    const faculty = await User.aggregate([
      { $match: { role: 'faculty' } },
      {
        $lookup: {
          from: 'timetables',
          localField: 'facultyName',
          foreignField: 'facultyName',
          as: 'timetableEntries'
        }
      },
      {
        $project: {
          _id: 1,
          email: 1,
          facultyName: 1,
          name: 1,
          // Extract unique subjects from timetable entries
          subjectsFromTimetable: {
            $cond: [
              { $gt: [{ $size: '$timetableEntries' }, 0] },
              {
                $reduce: {
                  input: '$timetableEntries',
                  initialValue: [],
                  in: {
                    $cond: [
                      { $in: ['$$this.subject', '$$value'] },
                      '$$value',
                      { $concatArrays: ['$$value', ['$$this.subject']] }
                    ]
                  }
                }
              },
              []
            ]
          }
        }
      },
      { $sort: { name: 1 } }
    ]);
    
    // Format the response
    const formattedFaculty = faculty.map(fac => ({
      _id: fac._id,
      email: fac.email,
      facultyName: fac.facultyName,
      name: fac.name,
      subjects: fac.subjectsFromTimetable || []
    }));
    
    res.json(formattedFaculty);
  } catch (error) {
    console.error('Get faculty error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * POST /api/admin/faculty
 * Add new faculty member
 */
router.post('/faculty', async (req, res) => {
  try {
    const { email, password, facultyName, subjects } = req.body;
    
    if (!email || !password || !facultyName) {
      return res.status(400).json({ message: 'Missing required fields' });
    }
    
    // Check if faculty already exists
    const existing = await User.findOne({ email });
    if (existing) {
      return res.status(409).json({ message: 'Faculty already exists with this email' });
    }
    
    const bcrypt = require('bcryptjs');
    const hashedPassword = await bcrypt.hash(password, 10);
    
    const faculty = new User({
      email,
      password: hashedPassword,
      facultyName,
      name: facultyName,
      subjects: subjects || [],
      role: 'faculty'
    });
    
    await faculty.save();
    
    res.json({
      success: true,
      message: 'Faculty added successfully',
      data: {
        _id: faculty._id,
        email: faculty.email,
        facultyName: faculty.facultyName,
        subjects: faculty.subjects
      }
    });
  } catch (error) {
    console.error('Add faculty error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * DELETE /api/admin/faculty/:id
 * Delete faculty member
 */
router.delete('/faculty/:id', async (req, res) => {
  try {
    const faculty = await User.findByIdAndDelete(req.params.id);
    
    if (!faculty) {
      return res.status(404).json({ message: 'Faculty not found' });
    }
    
    res.json({
      success: true,
      message: 'Faculty deleted successfully'
    });
  } catch (error) {
    console.error('Delete faculty error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * PUT /api/admin/faculty/:id
 * Update faculty information
 */
router.put('/faculty/:id', async (req, res) => {
  try {
    const { facultyName, name, subjects } = req.body;
    
    const updateData = {};
    if (facultyName !== undefined) updateData.facultyName = facultyName;
    if (name !== undefined) updateData.name = name;
    if (subjects !== undefined) updateData.subjects = subjects;
    
    const faculty = await User.findByIdAndUpdate(
      req.params.id,
      updateData,
      { new: true }
    ).select('-password');
    
    if (!faculty) {
      return res.status(404).json({ message: 'Faculty not found' });
    }
    
    res.json({
      success: true,
      message: 'Faculty updated successfully',
      data: faculty
    });
  } catch (error) {
    console.error('Update faculty error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * GET /api/admin/export-attendance
 * Export attendance records as Excel file
 */
router.get('/export-attendance', async (req, res) => {
  try {
    const { subject, year, batch, startDate, endDate } = req.query;
    const ExcelJS = require('exceljs');
    
    let query = {};
    if (subject) query.subject = subject;
    if (year) query.year = year;
    if (batch) query.batch = batch;
    if (startDate || endDate) {
      query.date = {};
      if (startDate) query.date.$gte = new Date(startDate);
      if (endDate) query.date.$lte = new Date(endDate);
    }
    
    // Use aggregation to JOIN with User collection to get current roll number and signature
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
          facultyName: 1,
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
    const headers = ['Roll Number', 'Student ID', 'Student Name', 'Subject', 'Lecture Type', 'Status', 'Marked By', 'Date', 'Signature'];
    
    // Set column widths first (more stable approach)
    worksheet.columns = [
      { header: 'Roll Number', width: 14 },
      { header: 'Student ID', width: 14 },
      { header: 'Student Name', width: 22 },
      { header: 'Subject', width: 22 },
      { header: 'Lecture Type', width: 15 },
      { header: 'Status', width: 10 },
      { header: 'Marked By', width: 18 },
      { header: 'Date', width: 14 },
      { header: 'Signature', width: 25 }
    ];
    
    const headerRow = worksheet.getRow(1);
    
    // Style header row with more compatible settings
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
        record.facultyName || '',
        record.date ? record.date.toISOString().split('T')[0] : '',
        record.studentSignatureInitials || ''
      ];

      const row = worksheet.addRow(rowData);

      // Style data rows
      row.eachCell((cell, colNumber) => {
        // Signature column - use Vladimir Script for elegant handwritten look  
        if (colNumber === 9) {
          cell.font = { name: 'Vladimir Script', size: 14, bold: true, color: { argb: 'FF000000' } };
        } else {
          cell.font = { name: 'Calibri', size: 11 };
        }

        cell.alignment = { 
          horizontal: colNumber === 9 ? 'center' : 'left', 
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
    res.status(500).json({ message: `Export failed: ${error.message}` });
  }
});

module.exports = router;