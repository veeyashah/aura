const mongoose = require('mongoose');
require('dotenv').config();

const dropUsernameIndex = async () => {
  try {
    // Connect to MongoDB
    await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/attendance_system', {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });
    
    console.log('✅ Connected to MongoDB');
    
    const db = mongoose.connection.db;
    const collection = db.collection('users');
    
    try {
      // Try to drop the username index if it exists
      await collection.dropIndex('username_1');
      console.log('✅ Successfully dropped username_1 index');
    } catch (error) {
      if (error.code === 27 || error.codeName === 'IndexNotFound') {
        console.log('ℹ️  username_1 index does not exist, skipping...');
      } else {
        throw error;
      }
    }
    
    // List all indexes to verify
    const indexes = await collection.indexes();
    console.log('\nCurrent indexes:', indexes.map(idx => idx.name));
    
    await mongoose.disconnect();
    console.log('\n✅ Disconnected from MongoDB');
    process.exit(0);
  } catch (error) {
    console.error('❌ Error:', error);
    process.exit(1);
  }
};

dropUsernameIndex();
