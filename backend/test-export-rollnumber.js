const mongoose = require('mongoose');
const Attendance = require('./models/Attendance');
const User = require('./models/User');
require('dotenv').config();

const testExport = async () => {
  try {
    await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/aura');
    console.log('✅ Database connected');

    // Fetch attendance records sorted by roll number
    const records = await Attendance.find({})
      .select('studentId studentName studentRollNumber studentSignatureInitials subject lectureType status markedBy date')
      .sort({ studentRollNumber: 1 })
      .lean();

    if (records.length === 0) {
      console.log('❌ No attendance records found');
      process.exit(0);
    }

    console.log(`\n✅ Found ${records.length} attendance records\n`);
    console.log('Sample records (sorted by roll number):');
    console.log('─'.repeat(100));
    
    records.slice(0, 5).forEach((record, idx) => {
      console.log(`${idx + 1}. Roll: ${record.studentRollNumber || 'N/A'} | Student: ${record.studentName} | Signature: "${record.studentSignatureInitials || 'EMPTY'}"`);
    });

    // Check if signatures are populated
    const withSignatures = records.filter(r => r.studentSignatureInitials && r.studentSignatureInitials.trim());
    const withRollNumbers = records.filter(r => r.studentRollNumber && r.studentRollNumber.trim());

    console.log('\n─'.repeat(100));
    console.log(`\nSignature Population:`);
    console.log(`  ✅ Total records: ${records.length}`);
    console.log(`  ${withSignatures.length > 0 ? '✅' : '❌'} Records with signatures: ${withSignatures.length}/${records.length}`);
    console.log(`  ${withRollNumbers.length > 0 ? '✅' : '❌'} Records with roll numbers: ${withRollNumbers.length}/${records.length}`);

    // Check sorting
    const rollNumbers = records.map(r => r.studentRollNumber).filter(Boolean);
    const isSorted = rollNumbers.every((val, i, arr) => i === 0 || val >= arr[i - 1]);
    console.log(`  ${isSorted ? '✅' : '❌'} Sorted by roll number ascending: ${isSorted}`);

    if (!isSorted && rollNumbers.length > 1) {
      console.log(`     Error: Roll numbers not in order`);
      console.log(`     First 5: ${rollNumbers.slice(0, 5).join(', ')}`);
    }

    console.log('\n✅ Export test complete');
    process.exit(0);
  } catch (error) {
    console.error('❌ Error:', error.message);
    process.exit(1);
  }
};

testExport();
