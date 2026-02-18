/**
 * Route Testing Script
 * Run this to verify all backend routes are working
 */

const axios = require('axios');

const BASE_URL = 'http://localhost:5000/api';
let authToken = '';

// Test credentials
const ADMIN_CREDS = {
  email: 'admin@attendance.com',
  password: 'admin123'
};

const FACULTY_CREDS = {
  email: 'aj@attendance.com',
  password: 'aj123'
};

async function testRoute(method, endpoint, data = null, token = null) {
  try {
    const config = {
      method,
      url: `${BASE_URL}${endpoint}`,
      headers: token ? { Authorization: `Bearer ${token}` } : {}
    };
    
    if (data) {
      config.data = data;
    }
    
    const response = await axios(config);
    console.log(`✅ ${method} ${endpoint} - Status: ${response.status}`);
    return response.data;
  } catch (error) {
    console.log(`❌ ${method} ${endpoint} - Error: ${error.response?.status || error.message}`);
    return null;
  }
}

async function runTests() {
  console.log('🧪 Starting Route Tests...\n');
  
  // Test 1: Admin Login
  console.log('📝 Test 1: Admin Login');
  const adminLogin = await testRoute('POST', '/auth/login', ADMIN_CREDS);
  const adminToken = adminLogin?.token;
  
  if (!adminToken) {
    console.log('❌ Admin login failed - cannot continue tests');
    return;
  }
  console.log('');
  
  // Test 2: Admin Routes
  console.log('📝 Test 2: Admin Routes');
  await testRoute('GET', '/admin/dashboard', null, adminToken);
  await testRoute('GET', '/admin/students', null, adminToken);
  await testRoute('GET', '/admin/faculty', null, adminToken);
  await testRoute('GET', '/admin/timetable?year=FY', null, adminToken);
  console.log('');
  
  // Test 3: Faculty Login
  console.log('📝 Test 3: Faculty Login');
  const facultyLogin = await testRoute('POST', '/auth/login', FACULTY_CREDS);
  const facultyToken = facultyLogin?.token;
  
  if (!facultyToken) {
    console.log('⚠️  Faculty login failed - create faculty first');
    console.log('');
  } else {
    console.log('');
    
    // Test 4: Faculty Routes
    console.log('📝 Test 4: Faculty Routes');
    await testRoute('GET', '/faculty/dashboard', null, facultyToken);
    await testRoute('GET', '/faculty/subjects', null, facultyToken);
    await testRoute('GET', '/faculty/timetable', null, facultyToken);
    await testRoute('GET', '/faculty/students?year=FY&batch=B1', null, facultyToken);
    await testRoute('GET', '/faculty/attendance', null, facultyToken);
    console.log('');
  }
  
  // Test 5: Python API
  console.log('📝 Test 5: Python API Health Check');
  try {
    const pythonResponse = await axios.get('http://localhost:8000/');
    console.log(`✅ Python API - Status: ${pythonResponse.status}`);
  } catch (error) {
    console.log(`❌ Python API - Not running or not accessible`);
  }
  console.log('');
  
  console.log('🎉 Route Tests Complete!\n');
  console.log('📊 Summary:');
  console.log('- If all tests show ✅, your backend is working correctly');
  console.log('- If you see ❌, check the error messages and fix the issues');
  console.log('- Make sure MongoDB is running before testing');
  console.log('- Make sure Python API is running for face recognition');
}

// Run tests
runTests().catch(console.error);
