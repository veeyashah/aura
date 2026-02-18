const mongoose = require('mongoose');

const timetableSchema = new mongoose.Schema({
  year: {
    type: String,
    enum: ['FY', 'SY', 'TY'],
    required: true,
  },
  day: {
    type: String,
    enum: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'],
    required: true,
  },
  timeSlot: {
    type: String,
    required: true,
  },
  facultyName: {
    type: String,
    required: true,
  },
  subject: {
    type: String,
    required: true,
  },
  lectureType: {
    type: String,
    enum: ['Lecture', 'Practical', 'Tutorial'],
    default: 'Lecture',
  },
  division: {
    type: String,
  },
  batch: {
    type: String,
    enum: ['B1', 'B2', 'ALL'],
  },
});

timetableSchema.index({ year: 1, day: 1 });

module.exports = mongoose.model('Timetable', timetableSchema);
