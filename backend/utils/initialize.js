const User = require('../models/User');
const Timetable = require('../models/Timetable');

const initializeDefaults = async () => {
  try {
    // Create or reset default admin
    let admin = await User.findOne({ email: 'admin@attendance.com' });
    if (!admin) {
      admin = new User({
        email: 'admin@attendance.com',
        password: 'admin123',
        role: 'admin',
      });
      await admin.save();
      console.log('✅ Default admin created: admin@attendance.com / admin123');
    } else {
      // Reset admin password to default if needed (uncomment to force reset)
      // admin.password = 'admin123';
      // await admin.save();
      // console.log('✅ Admin password reset to default');
      console.log('ℹ️  Admin user already exists');
      
      // Verify admin can login with default password
      const bcrypt = require('bcryptjs');
      const canLogin = await bcrypt.compare('admin123', admin.password);
      if (!canLogin) {
        console.log('⚠️  Admin password does not match default. Use admin@attendance.com / admin123');
      }
    }

    // Create specific 6 faculties
    const faculties = [
      { email: 'aj@attendance.com', password: 'aj123', facultyName: 'Anupama Jawale', subjects: [] },
      { email: 'rk@attendance.com', password: 'rk123', facultyName: 'Reeba Khan', subjects: [] },
      { email: 'rp@attendance.com', password: 'rp123', facultyName: 'Ruta Prabhu', subjects: [] },
      { email: 'sp@attendance.com', password: 'sp123', facultyName: 'Shweta Pawar', subjects: [] },
      { email: 'pj@attendance.com', password: 'pj123', facultyName: 'Prashant Jadhav', subjects: [] },
      { email: 'nk@attendance.com', password: 'nk123', facultyName: 'Neha KUSHE', subjects: [] },
    ];

    for (const facultyData of faculties) {
      const existing = await User.findOne({ email: facultyData.email });
      if (!existing) {
        const faculty = new User(facultyData);
        faculty.role = 'faculty';
        await faculty.save();
        console.log(`✅ Created faculty: ${facultyData.facultyName} (${facultyData.email} / ${facultyData.password})`);
      } else {
        // Update password if needed
        if (existing.password !== facultyData.password) {
          existing.password = facultyData.password;
          await existing.save();
          console.log(`✅ Updated password for: ${facultyData.facultyName}`);
        }
      }
    }
    console.log('✅ Faculty initialization complete');

    // Create default timetables if not exists
    const timetableExists = await Timetable.findOne();
    if (!timetableExists) {
      const timetables = [
        // FY Timetable
        { year: 'FY', day: 'Monday', timeSlot: '09:00-10:00', facultyName: 'Dr. Smith', subject: 'Mathematics', lectureType: 'Lecture', division: 'A', batch: 'B1' },
        { year: 'FY', day: 'Monday', timeSlot: '10:00-11:00', facultyName: 'Dr. Johnson', subject: 'Chemistry', lectureType: 'Lecture', division: 'A', batch: 'B1' },
        { year: 'FY', day: 'Tuesday', timeSlot: '09:00-10:00', facultyName: 'Dr. Williams', subject: 'Computer Science', lectureType: 'Lecture', division: 'A', batch: 'B1' },
        { year: 'FY', day: 'Tuesday', timeSlot: '10:00-11:00', facultyName: 'Dr. Brown', subject: 'English', lectureType: 'Lecture', division: 'A', batch: 'B1' },
        { year: 'FY', day: 'Wednesday', timeSlot: '09:00-10:00', facultyName: 'Dr. Smith', subject: 'Mathematics', lectureType: 'Practical', division: 'A', batch: 'B1' },
        { year: 'FY', day: 'Wednesday', timeSlot: '10:00-11:00', facultyName: 'Dr. Johnson', subject: 'Chemistry', lectureType: 'Practical', division: 'A', batch: 'B1' },
        { year: 'FY', day: 'Thursday', timeSlot: '09:00-10:00', facultyName: 'Dr. Williams', subject: 'Computer Science', lectureType: 'Practical', division: 'A', batch: 'B1' },
        { year: 'FY', day: 'Friday', timeSlot: '09:00-10:00', facultyName: 'Dr. Brown', subject: 'English', lectureType: 'Tutorial', division: 'A', batch: 'B1' },
        
        // SY Timetable
        { year: 'SY', day: 'Monday', timeSlot: '11:00-12:00', facultyName: 'Dr. Smith', subject: 'Mathematics', lectureType: 'Lecture', division: 'A', batch: 'B1' },
        { year: 'SY', day: 'Monday', timeSlot: '12:00-13:00', facultyName: 'Dr. Johnson', subject: 'Biology', lectureType: 'Lecture', division: 'A', batch: 'B1' },
        { year: 'SY', day: 'Tuesday', timeSlot: '11:00-12:00', facultyName: 'Dr. Williams', subject: 'Programming', lectureType: 'Lecture', division: 'A', batch: 'B1' },
        { year: 'SY', day: 'Wednesday', timeSlot: '11:00-12:00', facultyName: 'Dr. Davis', subject: 'History', lectureType: 'Lecture', division: 'A', batch: 'B1' },
        
        // TY Timetable
        { year: 'TY', day: 'Monday', timeSlot: '14:00-15:00', facultyName: 'Dr. Miller', subject: 'Economics', lectureType: 'Lecture', division: 'A', batch: 'B1' },
        { year: 'TY', day: 'Tuesday', timeSlot: '14:00-15:00', facultyName: 'Dr. Williams', subject: 'Programming', lectureType: 'Lecture', division: 'A', batch: 'B1' },
        { year: 'TY', day: 'Wednesday', timeSlot: '14:00-15:00', facultyName: 'Dr. Miller', subject: 'Business', lectureType: 'Lecture', division: 'A', batch: 'B1' },
      ];

      await Timetable.insertMany(timetables);
      console.log('✅ Default timetables created');
    }
  } catch (error) {
    console.error('❌ Error initializing defaults:', error);
  }
};

module.exports = { initializeDefaults };
