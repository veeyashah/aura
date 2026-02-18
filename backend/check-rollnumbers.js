const mongoose = require('mongoose');
const User = require('./models/User');
require('dotenv').config();

const checkRollNumbers = async () => {
  try {
    await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/aura');
    console.log('✅ Database connected\n');

    // Get all students
    const students = await User.find({ role: 'student' }).select('studentId name rollNumber signature');
    
    if (students.length === 0) {
      console.log('❌ No students found in database');
      process.exit(0);
    }

    console.log(`Found ${students.length} students:\n`);
    students.forEach((student, idx) => {
      console.log(`${idx + 1}. ID: ${student.studentId}`);
      console.log(`   Name: ${student.name}`);
      console.log(`   Roll: ${student.rollNumber || '❌ NOT SET'}`);
      console.log(`   Signature: ${student.signature ? '✅ YES' : '❌ NO'}\n`);
    });

    // Update students with roll numbers if missing
    const needsRollNumber = students.filter(s => !s.rollNumber);
    if (needsRollNumber.length > 0) {
      console.log(`\n⚠️  ${needsRollNumber.length} students missing roll numbers. Adding defaults...\n`);
      
      for (let i = 0; i < needsRollNumber.length; i++) {
        const rollNum = String(i + 1).padStart(4, '0'); // A001, A002, etc.
        await User.updateOne(
          { _id: needsRollNumber[i]._id },
          { rollNumber: `A${rollNum}` }
        );
        console.log(`  ✅ ${needsRollNumber[i].name} → A${rollNum}`);
      }
    }

    console.log('\n✅ Roll number check complete');
    process.exit(0);
  } catch (error) {
    console.error('❌ Error:', error.message);
    process.exit(1);
  }
};

checkRollNumbers();
