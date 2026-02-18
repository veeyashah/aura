'use client'

import { useEffect, useState } from 'react'
import Layout from '@/components/layout/Layout'
import api from '@/lib/api'
import { toast } from 'react-toastify'

export default function Timetable() {
  const [timetable, setTimetable] = useState<any[]>([])
  const [faculty, setFaculty] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedYear, setSelectedYear] = useState('FY')
  const [showAddModal, setShowAddModal] = useState(false)
  const [editingSlot, setEditingSlot] = useState<any>(null)
  const [formData, setFormData] = useState({
    year: 'FY',
    day: 'Monday',
    timeSlot: '09:00-10:00',
    facultyName: '',
    subject: '',
    lectureType: 'Lecture',
    division: 'A',
    batch: '', // No default batch
  })

  useEffect(() => {
    fetchTimetable()
    fetchFaculty()
  }, [selectedYear])

  const fetchTimetable = async () => {
    try {
      const response = await api.get(`/admin/timetable?year=${selectedYear}`)
      setTimetable(response.data)
    } catch (error: any) {
      toast.error('Failed to load timetable')
    } finally {
      setLoading(false)
    }
  }

  const fetchFaculty = async () => {
    try {
      const response = await api.get('/admin/faculty')
      setFaculty(response.data)
    } catch (error) {
      // Ignore
    }
  }

  const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
  const timeSlots = ['09:00-10:00', '10:00-11:00', '11:00-12:00', '12:00-13:00', '14:00-15:00', '15:00-16:00']

  const openAddModal = (day?: string, timeSlot?: string) => {
    setFormData({
      year: selectedYear,
      day: day || 'Monday',
      timeSlot: timeSlot || '09:00-10:00',
      facultyName: '',
      subject: '',
      lectureType: 'Lecture',
      division: 'A',
      batch: 'ALL', // Default to ALL for lectures
    })
    setEditingSlot(null)
    setShowAddModal(true)
  }

  const openEditModal = (slot: any) => {
    setFormData({
      year: slot.year,
      day: slot.day,
      timeSlot: slot.timeSlot,
      facultyName: slot.facultyName,
      subject: slot.subject,
      lectureType: slot.lectureType,
      division: slot.division || 'A',
      batch: slot.batch || (slot.lectureType === 'Lecture' ? 'ALL' : ''), // Default based on type
    })
    setEditingSlot(slot)
    setShowAddModal(true)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      if (editingSlot) {
        await api.put(`/admin/timetable/${editingSlot._id}`, formData)
        toast.success('Timetable entry updated successfully!')
      } else {
        await api.post('/admin/timetable', formData)
        toast.success('Timetable entry added successfully!')
      }
      setShowAddModal(false)
      fetchTimetable()
    } catch (error: any) {
      toast.error(error.response?.data?.message || 'Failed to save timetable entry')
    }
  }

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this timetable entry?')) return
    
    try {
      await api.delete(`/admin/timetable/${id}`)
      toast.success('Timetable entry deleted successfully!')
      fetchTimetable()
    } catch (error: any) {
      toast.error(error.response?.data?.message || 'Failed to delete timetable entry')
    }
  }

  return (
    <Layout role="admin">
      <div className="space-y-6">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 rounded-2xl p-8 shadow-2xl text-white">
          <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
            <div>
              <h1 className="text-4xl font-bold mb-2">📅 Timetable Management</h1>
              <p className="text-blue-100 text-lg">Manage class schedules and lecture slots</p>
            </div>
            <div className="flex gap-3">
              <select
                value={selectedYear}
                onChange={(e) => setSelectedYear(e.target.value)}
                className="px-6 py-3 border-2 border-white rounded-xl bg-white text-indigo-600 font-bold shadow-xl focus:ring-2 focus:ring-white transition-all"
              >
                <option value="FY">First Year (FY)</option>
                <option value="SY">Second Year (SY)</option>
                <option value="TY">Third Year (TY)</option>
              </select>
              <button
                onClick={() => openAddModal()}
                className="bg-white text-indigo-600 px-6 py-3 rounded-xl hover:bg-indigo-50 shadow-xl hover:shadow-2xl transition-all transform hover:-translate-y-0.5 font-bold whitespace-nowrap"
              >
                + Add Entry
              </button>
            </div>
          </div>
        </div>

        {loading ? (
          <div className="flex justify-center items-center py-16">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
          </div>
        ) : (
          <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="min-w-full">
                <thead className="bg-gradient-to-r from-slate-700 to-slate-800 text-white">
                  <tr>
                    <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider sticky left-0 bg-slate-800">Time</th>
                    {days.map((day) => (
                      <th key={day} className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider">
                        {day}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {timeSlots.map((timeSlot) => (
                    <tr key={timeSlot} className="hover:bg-gray-50 transition-colors">
                      <td className="px-6 py-4 whitespace-nowrap font-bold text-gray-800 bg-gradient-to-r from-gray-50 to-gray-100 sticky left-0">{timeSlot}</td>
                      {days.map((day) => {
                        const slot = timetable.find(
                          (t) => t.day === day && t.timeSlot === timeSlot && t.year === selectedYear
                        )
                        return (
                          <td key={day} className="px-6 py-4 min-w-[220px]">
                            {slot ? (
                              <div className="bg-gradient-to-br from-blue-50 to-indigo-50 border-2 border-blue-200 rounded-xl p-3 group relative hover:shadow-lg transition-all">
                                <div className="text-sm font-bold text-blue-900 mb-1">{slot.subject}</div>
                                <div className="text-xs text-blue-700 font-semibold">{slot.facultyName}</div>
                                <div className="flex gap-1 mt-2">
                                  <span className="text-xs bg-blue-200 text-blue-800 px-2 py-1 rounded-full font-semibold">{slot.lectureType}</span>
                                  <span className="text-xs bg-indigo-200 text-indigo-800 px-2 py-1 rounded-full font-semibold">{slot.batch}</span>
                                  <span className="text-xs bg-purple-200 text-purple-800 px-2 py-1 rounded-full font-semibold">{slot.division}</span>
                                </div>
                                <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 flex gap-1 transition-opacity">
                                  <button
                                    onClick={() => openEditModal(slot)}
                                    className="bg-blue-600 text-white px-3 py-1 rounded-lg text-xs hover:bg-blue-700 font-semibold shadow-lg"
                                  >
                                    Edit
                                  </button>
                                  <button
                                    onClick={() => handleDelete(slot._id)}
                                    className="bg-red-600 text-white px-3 py-1 rounded-lg text-xs hover:bg-red-700 font-semibold shadow-lg"
                                  >
                                    Delete
                                  </button>
                                </div>
                              </div>
                            ) : (
                              <button
                                onClick={() => openAddModal(day, timeSlot)}
                                className="w-full h-full min-h-[80px] border-2 border-dashed border-gray-300 rounded-xl p-3 text-gray-400 hover:border-indigo-400 hover:text-indigo-400 hover:bg-indigo-50 transition-all font-semibold"
                              >
                                + Add
                              </button>
                            )}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Add/Edit Modal */}
        {showAddModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full max-h-[90vh] overflow-y-auto">
              <div className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-t-2xl p-6 text-white sticky top-0">
                <h2 className="text-2xl font-bold">
                  {editingSlot ? '✏️ Edit Timetable Entry' : '➕ Add Timetable Entry'}
                </h2>
                <p className="text-blue-100 text-sm mt-1">Fill in the schedule details</p>
              </div>
              <form onSubmit={handleSubmit} className="p-6 space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-bold text-gray-700 mb-2">Year *</label>
                    <select
                      required
                      value={formData.year}
                      onChange={(e) => setFormData({ ...formData, year: e.target.value })}
                      className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
                    >
                      <option value="FY">FY</option>
                      <option value="SY">SY</option>
                      <option value="TY">TY</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-bold text-gray-700 mb-2">Day *</label>
                    <select
                      required
                      value={formData.day}
                      onChange={(e) => setFormData({ ...formData, day: e.target.value })}
                      className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
                    >
                      {days.map(day => <option key={day} value={day}>{day}</option>)}
                    </select>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-bold text-gray-700 mb-2">Time Slot *</label>
                    <select
                      required
                      value={formData.timeSlot}
                      onChange={(e) => setFormData({ ...formData, timeSlot: e.target.value })}
                      className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
                    >
                      {timeSlots.map(slot => <option key={slot} value={slot}>{slot}</option>)}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-bold text-gray-700 mb-2">Batch *</label>
                    <select
                      required
                      value={formData.batch}
                      onChange={(e) => setFormData({ ...formData, batch: e.target.value })}
                      className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
                    >
                      {formData.lectureType === 'Lecture' ? (
                        <option value="ALL">ALL</option>
                      ) : (
                        <>
                          <option value="">Select Batch</option>
                          <option value="B1">B1</option>
                          <option value="B2">B2</option>
                        </>
                      )}
                    </select>
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-bold text-gray-700 mb-2">Faculty *</label>
                  <select
                    required
                    value={formData.facultyName}
                    onChange={(e) => setFormData({ ...formData, facultyName: e.target.value })}
                    className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
                  >
                    <option value="">Select Faculty</option>
                    {faculty.map(f => (
                      <option key={f._id} value={f.facultyName}>{f.facultyName}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-bold text-gray-700 mb-2">Subject *</label>
                  <input
                    type="text"
                    required
                    value={formData.subject}
                    onChange={(e) => setFormData({ ...formData, subject: e.target.value })}
                    className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
                    placeholder="Enter subject name"
                  />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-bold text-gray-700 mb-2">Lecture Type *</label>
                    <select
                      required
                      value={formData.lectureType}
                      onChange={(e) => {
                        const newLectureType = e.target.value;
                        setFormData({ 
                          ...formData, 
                          lectureType: newLectureType,
                          batch: newLectureType === 'Lecture' ? 'ALL' : ''
                        });
                      }}
                      className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
                    >
                      <option value="Lecture">Lecture</option>
                      <option value="Practical">Practical</option>
                      <option value="Tutorial">Tutorial</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-bold text-gray-700 mb-2">Division</label>
                    <input
                      type="text"
                      value={formData.division}
                      onChange={(e) => setFormData({ ...formData, division: e.target.value })}
                      className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
                      placeholder="e.g., A"
                    />
                  </div>
                </div>
                <div className="flex gap-3 pt-4">
                  <button
                    type="submit"
                    className="flex-1 bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-3 rounded-xl hover:from-blue-700 hover:to-indigo-700 font-bold shadow-xl hover:shadow-2xl transition-all"
                  >
                    {editingSlot ? 'Update' : 'Add'} Entry
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowAddModal(false)}
                    className="flex-1 bg-gray-200 text-gray-700 py-3 rounded-xl hover:bg-gray-300 font-bold transition-all"
                  >
                    Cancel
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
      </div>
    </Layout>
  )
}
