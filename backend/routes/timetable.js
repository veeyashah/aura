const express = require('express');
const Timetable = require('../models/Timetable');
const { auth, authorize } = require('../middleware/auth');

const router = express.Router();
router.use(auth);

/**
 * GET /api/timetable
 * Get timetable (filtered based on user role)
 */
router.get('/', async (req, res) => {
  try {
    const { day, year, batch } = req.query;

    let query = {};
    if (day) query.day = day;
    if (year) query.year = year;
    if (batch && batch !== 'ALL') query.batch = batch;

    const records = await Timetable.find(query).sort({ day: 1, startTime: 1 });

    res.json({
      success: true,
      data: records
    });
  } catch (error) {
    console.error('Get timetable error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * POST /api/timetable
 * Create timetable (admin only)
 */
router.post('/', authorize('admin'), async (req, res) => {
  try {
    const { subject, faculty, day, startTime, endTime, lectureType, year, batch, room } = req.body;

    if (!subject || !faculty || !day || !startTime || !endTime) {
      return res.status(400).json({ message: 'Missing required fields' });
    }

    const timetable = new Timetable({
      subject,
      faculty,
      day,
      startTime,
      endTime,
      lectureType: lectureType || 'Lecture',
      year,
      batch,
      room
    });

    await timetable.save();

    res.json({
      success: true,
      message: 'Timetable created',
      data: timetable
    });
  } catch (error) {
    console.error('Create timetable error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

module.exports = router;
