const mongoose = require('mongoose');
const User = require('../models/User');
require('dotenv').config();

const resetAdmin = async () => {
  try {
    // Connect to MongoDB
    await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/attendance_system', {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });
    
    console.log('✅ Connected to MongoDB');
    
    // Find or create admin user
    let admin = await User.findOne({ email: 'admin@attendance.com' });
    
    if (admin) {
      // Reset password
      admin.password = 'admin123';
      await admin.save();
      console.log('✅ Admin password reset successfully');
      console.log('   Email: admin@attendance.com');
      console.log('   Password: admin123');
    } else {
      // Create admin user
      admin = new User({
        email: 'admin@attendance.com',
        password: 'admin123',
        role: 'admin',
      });
      await admin.save();
      console.log('✅ Admin user created successfully');
      console.log('   Email: admin@attendance.com');
      console.log('   Password: admin123');
    }
    
    await mongoose.disconnect();
    console.log('✅ Disconnected from MongoDB');
    process.exit(0);
  } catch (error) {
    console.error('❌ Error:', error);
    process.exit(1);
  }
};

resetAdmin();
