"use client"

import { useState } from 'react'
import Layout from '@/components/layout/Layout'
import api from '@/lib/api'
import { toast } from 'react-toastify'

export default function ExportAttendance() {
  const [lectureFilters, setLectureFilters] = useState({
    startDate: '',
    endDate: '',
    facultyName: '',
    subject: '',
    year: '',
    batch: '',
  })
  const [loading, setLoading] = useState(false)

  const handleLectureExport = async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams()
      Object.entries(lectureFilters).forEach(([key, value]) => {
        if (value) params.append(key, value)
      })

      const response = await api.get(`/admin/export-attendance?${params.toString()}`, {
        responseType: 'blob',
        headers: { 'Accept': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' }
      })

      const blob = new Blob([response.data], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' })
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `lecture_attendance_${new Date().toISOString().split('T')[0]}.xlsx`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)

      toast.success('Lecture attendance Excel exported successfully!')
    } catch (error: any) {
      console.error('Export error:', error)
      toast.error(error.response?.data?.message || error.message || 'Export failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Layout role="admin">
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 rounded-2xl p-8 shadow-2xl text-white">
          <h1 className="text-4xl font-bold mb-2">📊 Export Attendance Data</h1>
          <p className="text-emerald-100 text-lg">Download attendance reports in Excel format</p>
        </div>

        <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden p-8">
          <div className="mb-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-2">📚 Export Lecture Attendance</h2>
            <p className="text-gray-600">Filter and download lecture attendance records</p>
          </div>

          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-bold text-gray-700 mb-2">Start Date</label>
                <input
                  type="date"
                  value={lectureFilters.startDate}
                  onChange={(e) => setLectureFilters({ ...lectureFilters, startDate: e.target.value })}
                  className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
                />
              </div>
              <div>
                <label className="block text-sm font-bold text-gray-700 mb-2">End Date</label>
                <input
                  type="date"
                  value={lectureFilters.endDate}
                  onChange={(e) => setLectureFilters({ ...lectureFilters, endDate: e.target.value })}
                  className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-bold text-gray-700 mb-2">Faculty Name</label>
              <input
                type="text"
                value={lectureFilters.facultyName}
                onChange={(e) => setLectureFilters({ ...lectureFilters, facultyName: e.target.value })}
                placeholder="Leave empty for all faculty"
                className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
              />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-bold text-gray-700 mb-2">Subject</label>
                <input
                  type="text"
                  value={lectureFilters.subject}
                  onChange={(e) => setLectureFilters({ ...lectureFilters, subject: e.target.value })}
                  placeholder="Leave empty for all subjects"
                  className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
                />
              </div>
              <div>
                <label className="block text-sm font-bold text-gray-700 mb-2">Year</label>
                <select
                  value={lectureFilters.year}
                  onChange={(e) => setLectureFilters({ ...lectureFilters, year: e.target.value })}
                  className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
                >
                  <option value="">All Years</option>
                  <option value="FY">First Year (FY)</option>
                  <option value="SY">Second Year (SY)</option>
                  <option value="TY">Third Year (TY)</option>
                </select>
              </div>
            </div>

            <div>
              <label className="block text-sm font-bold text-gray-700 mb-2">Batch</label>
              <select
                value={lectureFilters.batch}
                onChange={(e) => setLectureFilters({ ...lectureFilters, batch: e.target.value })}
                className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
              >
                <option value="">All Batches</option>
                <option value="B1">Batch 1 (B1)</option>
                <option value="B2">Batch 2 (B2)</option>
              </select>
            </div>

            <button
              onClick={handleLectureExport}
              disabled={loading}
              className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-4 px-6 rounded-xl hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 shadow-xl hover:shadow-2xl transition-all font-bold text-lg"
            >
              {loading ? 'Exporting...' : '📥 Export Lecture Attendance Excel'}
            </button>
          </div>
        </div>
      </div>
    </Layout>
  )
}
