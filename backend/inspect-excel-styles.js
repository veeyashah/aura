const XLSX = require('xlsx');
const fs = require('fs');

const inspectExcelStyles = () => {
  try {
    // Read the test export file
    const filePath = 'test-attendance-export.xlsx';
    const workbook = XLSX.readFile(filePath);
    const worksheet = workbook.Sheets['Attendance'];

    console.log('✅ Excel file loaded: test-attendance-export.xlsx\n');

    // Check signature column (I) cells for styles
    console.log('Signature Column (I) Cell Analysis:');
    console.log('─'.repeat(80));

    for (let r = 0; r < 5; r++) {
      const cellAddress = XLSX.utils.encode_cell({ c: 8, r }); // Column I = index 8
      const cell = worksheet[cellAddress];
      
      if (cell) {
        console.log(`Cell ${cellAddress}:`);
        console.log(`  Value: ${cell.v}`);
        console.log(`  Type: ${cell.t}`);
        console.log(`  Style (s): ${JSON.stringify(cell.s || 'NOT SET')}`);
        console.log(`  Format (z): ${cell.z || 'NOT SET'}`);
        console.log('');
      }
    }

    // List all sheets and their properties
    console.log('Workbook Structure:');
    console.log(`  Sheets: ${Object.keys(workbook.Sheets).join(', ')}`);
    console.log(`  Props: ${JSON.stringify(workbook.Props || {})}`);

    // Check for theme and style info
    if (workbook.SSF) {
      console.log(`  Number Formats defined: ${Object.keys(workbook.SSF).length}`);
    }

    console.log('\n' + '─'.repeat(80));
    console.log('\n⚠️  Note: XLSX library has limited style support.');
    console.log('   Font styling may require opening and resaving in Excel.');
    console.log('   Alternative: Use a more robust Excel library like ExcelJS\n');

  } catch (error) {
    console.error('❌ Error:', error.message);
  }
};

inspectExcelStyles();
