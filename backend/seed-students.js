// Seed test students into database
const mongoose = require('mongoose');
const User = require('./models/User');
const bcrypt = require('bcrypt');

async function seedStudents() {
  try {
    const mongoUri = process.env.MONGODB_URI || 'mongodb://localhost:27017/aura';
    await mongoose.connect(mongoUri);
    console.log('✅ Connected to MongoDB\n');

    // Clear existing students
    const deleteResult = await User.deleteMany({ role: 'student' });
    console.log(`🗑️ Deleted ${deleteResult.deletedCount} existing students\n`);

    // Create test students
    const studentsData = [
      { studentId: 'A015', name: 'Dhriti Panchal', email: 'dhriti@example.com', year: 'FY', batch: 'B1', rollNumber: 'A015' },
      { studentId: 'A024', name: 'Veeya Shah', email: 'veeya@example.com', year: 'FY', batch: 'B2', rollNumber: 'A024' },
    ];

    const hashedPassword = await bcrypt.hash('password123', 10);

    const createdStudents = await User.insertMany(
      studentsData.map(s => ({
        ...s,
        password: hashedPassword,
        role: 'student',
        isTrained: false,
        faceEmbeddings: [],  // Empty initially
        signatureSubmitted: false
      }))
    );

    console.log(`✅ Created ${createdStudents.length} test students:\n`);
    createdStudents.forEach(s => {
      console.log(`   📌 ${s.studentId} - ${s.name} (${s.year}, Batch ${s.batch})`);
    });

    console.log(`\n🎓 Students are now ready to train their faces!`);
    console.log(`📝 Login with:`);
    console.log(`   Student ID: A015 (or A024, A025, A026)`);
    console.log(`   Password: password123`);

    await mongoose.connection.close();
    process.exit(0);
  } catch (error) {
    console.error('❌ Error:', error.message);
    process.exit(1);
  }
}

seedStudents();
