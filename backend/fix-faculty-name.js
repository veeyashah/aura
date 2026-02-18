/**
 * Fix Faculty Name Script
 * Updates faculty name to match what's used in timetable
 */

const mongoose = require('mongoose');
const User = require('./models/User');

async function fixFacultyName() {
  try {
    await mongoose.connect('mongodb://127.0.0.1:27017/attendance_system');
    console.log('✅ Connected to MongoDB');

    // Find the faculty user
    const faculty = await User.findOne({ email: 'aj@attendance.com' });
    
    if (!faculty) {
      console.log('❌ Faculty not found with email aj@attendance.com');
      process.exit(1);
    }

    console.log(`Current faculty name: ${faculty.facultyName || faculty.name}`);
    
    // Update to simple name "AJ"
    faculty.facultyName = 'AJ';
    faculty.name = 'AJ';
    await faculty.save();
    
    console.log('✅ Faculty name updated to: AJ');
    console.log('\nNow you can:');
    console.log('1. Login as admin');
    console.log('2. Go to Timetable Management');
    console.log('3. Add entries with Faculty: AJ');
    console.log('4. Login as faculty and dashboard will work!');
    
    await mongoose.connection.close();
  } catch (error) {
    console.error('❌ Error:', error);
    process.exit(1);
  }
}

fixFacultyName();
