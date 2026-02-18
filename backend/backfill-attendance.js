const mongoose = require('mongoose');
const Attendance = require('./models/Attendance');
const User = require('./models/User');
require('dotenv').config();

const backfillSignaturesAndRollNumbers = async () => {
  try {
    await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/aura');
    console.log('✅ Database connected\n');

    // Get all attendance records
    const records = await Attendance.find({});
    console.log(`Found ${records.length} attendance records to process\n`);

    let updated = 0;
    let errors = 0;

    for (const record of records) {
      try {
        // Fetch the student
        const student = await User.findOne({ studentId: record.studentId });
        
        if (!student) {
          console.log(`⚠️  Student ${record.studentId} not found, skipping`);
          continue;
        }

        let changes = {};

        // Extract signature initials if the record doesn't have them
        if (!record.studentSignatureInitials && student.signature) {
          try {
            const parsed = JSON.parse(student.signature);
            changes.studentSignatureInitials = parsed.initials || '';
          } catch (e) {
            changes.studentSignatureInitials = student.signature || '';
          }
        }

        // Set roll number if the record doesn't have it
        if (!record.studentRollNumber && student.rollNumber) {
          changes.studentRollNumber = student.rollNumber;
        }

        // Update if there are changes
        if (Object.keys(changes).length > 0) {
          await Attendance.updateOne({ _id: record._id }, changes);
          updated++;
          console.log(`✅ Updated: ${record.studentName} - Roll: ${changes.studentRollNumber || record.studentRollNumber} | Sig: ${changes.studentSignatureInitials || record.studentSignatureInitials}`);
        }
      } catch (err) {
        errors++;
        console.error(`❌ Error updating record: ${err.message}`);
      }
    }

    console.log(`\n✅ Backfill complete`);
    console.log(`   Updated: ${updated}/${records.length}`);
    console.log(`   Errors: ${errors}`);
    process.exit(0);
  } catch (error) {
    console.error('❌ Error:', error.message);
    process.exit(1);
  }
};

backfillSignaturesAndRollNumbers();
