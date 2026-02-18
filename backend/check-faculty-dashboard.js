/**
 * Check Faculty Dashboard Data
 * Run this to see what data exists for faculty dashboard
 */

const mongoose = require('mongoose');
const User = require('./models/User');
const Timetable = require('./models/Timetable');
const Attendance = require('./models/Attendance');

async function checkDashboard() {
  try {
    await mongoose.connect('mongodb://127.0.0.1:27017/attendance_system');
    console.log('✅ Connected to MongoDB\n');

    // Find faculty user
    const faculty = await User.findOne({ email: 'aj@attendance.com' });
    
    if (!faculty) {
      console.log('❌ Faculty not found with email aj@attendance.com');
      process.exit(1);
    }

    console.log('📋 Faculty Info:');
    console.log('  Email:', faculty.email);
    console.log('  Name:', faculty.name);
    console.log('  Faculty Name:', faculty.facultyName);
    console.log('  Role:', faculty.role);
    console.log('');

    // Check timetable entries
    const facultyName = faculty.facultyName || faculty.name;
    console.log(`🔍 Looking for timetable entries with facultyName: "${facultyName}"`);
    
    const timetableEntries = await Timetable.find({ facultyName: facultyName });
    console.log(`  Found ${timetableEntries.length} timetable entries`);
    
    if (timetableEntries.length > 0) {
      console.log('\n  Entries:');
      timetableEntries.forEach((entry, idx) => {
        console.log(`  ${idx + 1}. ${entry.day} ${entry.timeSlot} - ${entry.subject} (${entry.lectureType})`);
      });
    } else {
      console.log('  ⚠️  No timetable entries found!');
      console.log('\n  Checking all timetable entries:');
      const allEntries = await Timetable.find();
      console.log(`  Total timetable entries in DB: ${allEntries.length}`);
      if (allEntries.length > 0) {
        console.log('\n  Sample entries:');
        allEntries.slice(0, 3).forEach((entry, idx) => {
          console.log(`  ${idx + 1}. Faculty: "${entry.facultyName}" - ${entry.subject}`);
        });
        console.log('\n  ❌ PROBLEM: Faculty name mismatch!');
        console.log(`     Faculty user has: "${facultyName}"`);
        console.log(`     Timetable has: "${allEntries[0].facultyName}"`);
      }
    }
    console.log('');

    // Get distinct subjects
    const subjects = await Timetable.distinct('subject', { facultyName: facultyName });
    console.log(`📚 Subjects for this faculty: ${subjects.length}`);
    if (subjects.length > 0) {
      subjects.forEach((subject, idx) => {
        console.log(`  ${idx + 1}. ${subject}`);
      });
    }
    console.log('');

    // Check attendance
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const todayAttendance = await Attendance.countDocuments({
      markedBy: faculty.email,
      createdAt: { $gte: today }
    });
    console.log(`📊 Today's Attendance: ${todayAttendance} records`);
    console.log('');

    // Dashboard response
    console.log('✅ Dashboard Response:');
    console.log(JSON.stringify({
      todayActivity: todayAttendance,
      stats: subjects.map(s => ({ subject: s, total: 0, present: 0, percentage: '0.0' })),
      facultyName: facultyName
    }, null, 2));

    await mongoose.connection.close();
  } catch (error) {
    console.error('❌ Error:', error);
    process.exit(1);
  }
}

checkDashboard();
