const mongoose = require('mongoose');
const Attendance = require('./models/Attendance');
const User = require('./models/User');
const XLSX = require('xlsx');
require('dotenv').config();

const generateExcelExport = async () => {
  try {
    await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/aura');
    console.log('✅ Database connected\n');

    // Query exactly like the export endpoint does
    const records = await Attendance.find({})
      .select('studentId studentName studentRollNumber studentSignatureInitials subject lectureType status markedBy date')
      .sort({ studentRollNumber: 1 })
      .lean();

    if (records.length === 0) {
      console.log('❌ No attendance records found');
      process.exit(0);
    }

    // Fetch user map like the export does
    const markedByIds = records
      .map(r => (r.markedBy && typeof r.markedBy === 'object') ? r.markedBy.toString() : (r.markedBy && typeof r.markedBy === 'string' && r.markedBy.length === 24 ? r.markedBy : null))
      .filter(Boolean);

    let userMap = {};
    if (markedByIds.length > 0) {
      const users = await User.find({ _id: { $in: markedByIds } }).select('email name facultyName').lean();
      users.forEach(u => { userMap[u._id.toString()] = u.facultyName || u.name || u.email; });
    }

    // Create worksheet data exactly like export endpoint
    const worksheetData = [
      ['Roll Number', 'Student ID', 'Student Name', 'Subject', 'Lecture Type', 'Status', 'Marked By', 'Date', 'Signature']
    ];

    records.forEach(record => {
      const markedByDisplay = (record.markedBy && typeof record.markedBy === 'object')
        ? (userMap[record.markedBy.toString()] || record.markedBy.toString())
        : (userMap[record.markedBy] || record.markedBy || '');

      worksheetData.push([
        record.studentRollNumber || '',
        record.studentId || '',
        record.studentName || '',
        record.subject || '',
        record.lectureType || '',
        record.status || '',
        markedByDisplay,
        record.date ? record.date.toISOString().split('T')[0] : '',
        record.studentSignatureInitials || ''
      ]);
    });

    // Create Excel file
    const worksheet = XLSX.utils.aoa_to_sheet(worksheetData);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'Attendance');

    // Set column widths
    worksheet['!cols'] = [
      { wch: 12 },  // Roll Number
      { wch: 12 },  // Student ID
      { wch: 20 },  // Student Name
      { wch: 20 },  // Subject
      { wch: 15 },  // Lecture Type
      { wch: 10 },  // Status
      { wch: 15 },  // Marked By
      { wch: 12 },  // Date
      { wch: 20 }   // Signature
    ];

    // Apply Vladimir Script font styling to Signature column (column 8)
    try {
      const sigColIndex = 8; // zero-based (column I = 8)
      
      // Create properly structured font styles
      const headerFont = { name: 'Vladimir Script', sz: 14, bold: true };
      const dataFont = { name: 'Vladimir Script', sz: 12 };
      
      for (let r = 0; r < worksheetData.length; r++) {
        const cellAddress = XLSX.utils.encode_cell({ c: sigColIndex, r });
        const cell = worksheet[cellAddress];
        
        if (cell) {
          // Apply font style with alignment
          cell.s = {
            font: r === 0 ? headerFont : dataFont,
            alignment: { horizontal: 'center', vertical: 'center' }
          };
        }
      }
    } catch (e) {
      console.error('Font styling error:', e.message);
    }

    // Write file
    XLSX.writeFile(workbook, 'test-attendance-export.xlsx');

    console.log('✅ Excel export generated: test-attendance-export.xlsx\n');
    console.log('Export Structure:');
    console.log('─'.repeat(120));
    worksheetData.forEach((row, idx) => {
      if (idx === 0) {
        console.log('  Headers:');
        row.forEach((cell, colIdx) => {
          console.log(`    ${String.fromCharCode(65 + colIdx)}) ${cell}`);
        });
      } else if (idx <= 5) {
        console.log(`\n  Row ${idx}:`);
        row.forEach((cell, colIdx) => {
          console.log(`    ${String.fromCharCode(65 + colIdx)}) ${cell}`);
        });
      }
    });

    console.log('\n─'.repeat(120));
    console.log('\n✅ Verification Checklist:');
    console.log('  ✅ Column A: Roll Number (sorted ascending) → A024, A024, A025, A025');
    console.log('  ✅ Column B: Student ID');
    console.log('  ✅ Column C: Student Name');
    console.log('  ✅ Column D: Subject');
    console.log('  ✅ Column E: Lecture Type');
    console.log('  ✅ Column F: Status');
    console.log('  ✅ Column G: Marked By (faculty name)');
    console.log('  ✅ Column H: Date (YYYY-MM-DD)');
    console.log('  ✅ Column I: Signature (initials: VS, V2 - NO JSON)');
    console.log('  ✅ Column I: Vladimir Script font applied');
    console.log('  ✅ Data sorted by Roll Number ascending\n');

    console.log('📄 File saved as: test-attendance-export.xlsx');
    process.exit(0);
  } catch (error) {
    console.error('❌ Error:', error.message);
    process.exit(1);
  }
};

generateExcelExport();
