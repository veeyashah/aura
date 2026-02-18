const XLSX = require('xlsx');
const path = 'C:/aura/tmp_attendance_export.xlsx';
try{
  const wb = XLSX.readFile(path, { cellStyles: true });
  const sheetName = wb.SheetNames[0];
  const ws = wb.Sheets[sheetName];
  console.log('Sheet:', sheetName);
  const range = XLSX.utils.decode_range(ws['!ref']);
  const rows = [];
  for(let R = range.s.r; R <= Math.min(range.e.r, range.s.r+20); ++R){
    const row = [];
    for(let C = range.s.c; C <= range.e.c; ++C){
      const addr = XLSX.utils.encode_cell({r:R,c:C});
      const cell = ws[addr];
      if(C === 8){ // signature column index
        console.log('Row', R+1, 'cell:', addr, cell ? (cell.v ? cell.v : '<empty>') : '<no cell>', 'style=', cell && cell.s ? JSON.stringify(cell.s) : 'none');
      }
      row.push(cell ? cell.v : '');
    }
    rows.push(row);
  }
  console.log('\nFirst rows preview:');
  console.table(rows);
}catch(e){
  console.error('Failed to read xlsx:', e);
  process.exit(1);
}
