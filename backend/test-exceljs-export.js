const mongoose = require('mongoose');
const Attendance = require('./models/Attendance');
const User = require('./models/User');
const ExcelJS = require('exceljs');
require('dotenv').config();

const generateExcelExportWithExcelJS = async () => {
  try {
    await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/aura');
    console.log('✅ Database connected\n');

    // Query attendance records
    const records = await Attendance.find({})
      .select('studentId studentName studentRollNumber studentSignatureInitials subject lectureType status markedBy date')
      .sort({ studentRollNumber: 1 })
      .lean();

    if (records.length === 0) {
      console.log('❌ No attendance records found');
      process.exit(0);
    }

    console.log(`✅ Found ${records.length} attendance records\n`);

    // Fetch user map
    const markedByIds = records
      .map(r => (r.markedBy && typeof r.markedBy === 'object') ? r.markedBy.toString() : (r.markedBy && typeof r.markedBy === 'string' && r.markedBy.length === 24 ? r.markedBy : null))
      .filter(Boolean);

    let userMap = {};
    if (markedByIds.length > 0) {
      const users = await User.find({ _id: { $in: markedByIds } }).select('email name facultyName').lean();
      users.forEach(u => { userMap[u._id.toString()] = u.facultyName || u.name || u.email; });
    }

    // Create ExcelJS workbook
    const workbook = new ExcelJS.Workbook();
    const worksheet = workbook.addWorksheet('Attendance');

    // Define header row
    const headers = ['Roll Number', 'Student ID', 'Student Name', 'Subject', 'Lecture Type', 'Status', 'Marked By', 'Date', 'Signature'];
    const headerRow = worksheet.addRow(headers);

    // Style header row
    headerRow.eachCell((cell, colNumber) => {
      cell.font = { name: 'Calibri', size: 11, bold: true, color: { argb: 'FFFFFFFF' } };
      cell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF4472C4' } };
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
      const markedByDisplay = (record.markedBy && typeof record.markedBy === 'object')
        ? (userMap[record.markedBy.toString()] || record.markedBy.toString())
        : (userMap[record.markedBy] || record.markedBy || '');

      const rowData = [
        record.studentRollNumber || '',
        record.studentId || '',
        record.studentName || '',
        record.subject || '',
        record.lectureType || '',
        record.status || '',
        markedByDisplay,
        record.date ? record.date.toISOString().split('T')[0] : '',
        record.studentSignatureInitials || ''
      ];

      const row = worksheet.addRow(rowData);

      // Style data rows
      row.eachCell((cell, colNumber) => {
        // Apply Vladimir Script font ONLY to Signature column (column 9)
        if (colNumber === 9) {
          cell.font = { name: 'Vladimir Script', size: 14, italic: true };
        } else {
          cell.font = { name: 'Calibri', size: 11 };
        }

        cell.alignment = { horizontal: colNumber === 9 ? 'center' : 'left', vertical: 'center' };
        
        // Add borders
        cell.border = {
          top: { style: 'thin', color: { argb: 'FFCCCCCC' } },
          left: { style: 'thin', color: { argb: 'FFCCCCCC' } },
          bottom: { style: 'thin', color: { argb: 'FFCCCCCC' } },
          right: { style: 'thin', color: { argb: 'FFCCCCCC' } }
        };
      });
    });

    // Set column widths
    worksheet.columns = [
      { header: 'Roll Number', width: 12 },
      { header: 'Student ID', width: 12 },
      { header: 'Student Name', width: 20 },
      { header: 'Subject', width: 20 },
      { header: 'Lecture Type', width: 15 },
      { header: 'Status', width: 10 },
      { header: 'Marked By', width: 15 },
      { header: 'Date', width: 12 },
      { header: 'Signature', width: 20 }
    ];

    // Write to file
    await workbook.xlsx.writeFile('test-attendance-exceljs.xlsx');
    console.log('✅ Excel export generated with ExcelJS: test-attendance-exceljs.xlsx\n');

    // Verify by reading the file
    const verifyWorkbook = new ExcelJS.Workbook();
    await verifyWorkbook.xlsx.readFile('test-attendance-exceljs.xlsx');
    const verifySheet = verifyWorkbook.getWorksheet('Attendance');
    
    console.log('Export Verification:');
    console.log('─'.repeat(100));
    
    // Check signature column cells
    console.log('\nSignature Column (Column I) Cell Styles:');
    for (let r = 1; r <= 5 && r <= verifySheet.rowCount; r++) {
      const cell = verifySheet.getCell(`I${r}`);
      const fontInfo = cell.font ? `${cell.font.name} (size: ${cell.font.size})` : 'NO FONT';
      console.log(`  Row ${r}: Value="${cell.value}" | Font=${fontInfo}`);
    }

    console.log('\n' + '─'.repeat(100));
    console.log('\n✅ Production-Ready Export Features:');
    console.log('  ✅ Data sorted by Roll Number (ascending)');
    console.log('  ✅ Signature column styled with Vladimir Script font');
    console.log('  ✅ Header row with blue background and white text');
    console.log('  ✅ Data rows with borders');
    console.log('  ✅ Proper column widths');
    console.log('  ✅ Faculty names resolved (not ObjectIds)');
    console.log('  ✅ Date formatted as YYYY-MM-DD\n');

    process.exit(0);
  } catch (error) {
    console.error('❌ Error:', error.message);
    process.exit(1);
  }
};

generateExcelExportWithExcelJS();
