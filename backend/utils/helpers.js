const generateStudentCredentials = (studentId, studentName) => {
  // Generate username: student_<studentId>
  const username = `student_${studentId}`;
  
  // Generate password: <studentName>123 (remove spaces, lowercase)
  const cleanName = studentName.toLowerCase().replace(/\s+/g, '');
  const password = `${cleanName}123`;
  
  return { username, password };
};

module.exports = {
  generateStudentCredentials,
};
