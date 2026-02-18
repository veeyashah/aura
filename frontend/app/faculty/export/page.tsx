"use client"

import { useState, useEffect } from 'react'
import Layout from '@/components/layout/Layout'
import api from '@/lib/api'
import { toast } from 'react-toastify'

export default function ExportAttendance() {
  const [subjects, setSubjects] = useState<string[]>([])
  const [filters, setFilters] = useState({
    startDate: '',
    endDate: '',
    subject: '',
  })
  const [loading, setLoading] = useState(false)
  const [subjectsLoading, setSubjectsLoading] = useState(true)

  useEffect(() => {
    fetchSubjects()
  }, [])

  const fetchSubjects = async () => {
    try {
      const response = await api.get('/faculty/subjects')
      setSubjects(Array.isArray(response.data) ? response.data : [])
    } catch (error: any) {
      toast.error('Failed to load subjects')
      setSubjects([])
    } finally {
      setSubjectsLoading(false)
    }
  }

  const handleExport = async () => {
    // Validate that at least one filter is provided
    if (!filters.startDate && !filters.endDate && !filters.subject) {
      toast.warning('Please select at least one filter (date range or subject)')
      return
    }

    setLoading(true)
    try {
      const params = new URLSearchParams()
      if (filters.startDate) params.append('startDate', filters.startDate)
      if (filters.endDate) params.append('endDate', filters.endDate)
      if (filters.subject) params.append('subject', filters.subject)

      const response = await api.get(`/faculty/export-attendance?${params.toString()}`, {
        responseType: 'blob',
        headers: { 'Accept': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' }
      })

      const blob = new Blob([response.data], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' })
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `attendance_${new Date().toISOString().split('T')[0]}.xlsx`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)

      toast.success('Attendance exported successfully!')
    } catch (error: any) {
      console.error('Export error:', error)
      if (error.response?.status === 404) {
        toast.error('No attendance records found for the selected filters')
      } else {
        toast.error(error.response?.data?.message || error.message || 'Export failed')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <Layout role="faculty">
      <div className="space-y-6">
        {/* Header */}
        <div className="bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 rounded-2xl p-8 shadow-2xl text-white">
          <h1 className="text-4xl font-bold mb-2">📊 Export Attendance</h1>
          <p className="text-emerald-100 text-lg">Download your attendance records in Excel format with professional signatures</p>
        </div>

        {/* Export Card */}
        <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden p-8">
          <div className="mb-8">
            <h2 className="text-2xl font-bold text-gray-800 mb-2">📋 My Attendance Records</h2>
            <p className="text-gray-600">Filter and download your attendance data with signatures in Vladimir Script format</p>
          </div>

          <div className="space-y-6">
            {/* Filters Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Date Range Section */}
              <div className="space-y-4">
                <h3 className="text-lg font-bold text-gray-800">📅 Date Range (Optional)</h3>
                <div>
                  <label className="block text-sm font-bold text-gray-700 mb-2">Start Date</label>
                  <input
                    type="date"
                    value={filters.startDate}
                    onChange={(e) => setFilters({ ...filters, startDate: e.target.value })}
                    className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 transition-all"
                  />
                </div>
                <div>
                  <label className="block text-sm font-bold text-gray-700 mb-2">End Date</label>
                  <input
                    type="date"
                    value={filters.endDate}
                    onChange={(e) => setFilters({ ...filters, endDate: e.target.value })}
                    className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 transition-all"
                  />
                </div>
              </div>

              {/* Subject Section */}
              <div className="space-y-4">
                <h3 className="text-lg font-bold text-gray-800">📚 Subject (Optional)</h3>
                {subjectsLoading ? (
                  <div className="flex items-center justify-center py-8">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-emerald-600"></div>
                  </div>
                ) : subjects.length > 0 ? (
                  <div>
                    <select
                      value={filters.subject}
                      onChange={(e) => setFilters({ ...filters, subject: e.target.value })}
                      className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 transition-all"
                    >
                      <option value="">All Subjects</option>
                      {subjects.map((subject: string) => (
                        <option key={subject} value={subject}>
                          {subject}
                        </option>
                      ))}
                    </select>
                  </div>
                ) : (
                  <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <p className="text-yellow-800 text-sm">No subjects assigned yet. Contact admin to assign subjects.</p>
                  </div>
                )}
              </div>
            </div>

            {/* Info Box */}
            <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
              <p className="text-blue-800 text-sm">
                ℹ️ <strong>Note:</strong> Signatures will be displayed in Vladimir Script font for an elegant, professional appearance. Select at least one filter to export.
              </p>
            </div>

            {/* Export Button */}
            <div className="flex gap-3 pt-4">
              <button
                onClick={handleExport}
                disabled={loading}
                className="flex-1 bg-gradient-to-r from-emerald-600 to-teal-600 text-white py-3 px-6 rounded-xl hover:from-emerald-700 hover:to-teal-700 font-bold shadow-lg hover:shadow-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105 active:scale-95"
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Exporting...
                  </span>
                ) : (
                  '📥 Download Excel'
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Help Section */}
        <div className="bg-gradient-to-br from-teal-50 to-cyan-50 rounded-2xl p-6 border border-teal-200">
          <h3 className="text-lg font-bold text-teal-900 mb-3">❓ How to use</h3>
          <ul className="text-teal-800 space-y-2 text-sm">
            <li>✓ Select one or more filters to refine your attendance records</li>
            <li>✓ The exported Excel file includes Roll Number, Student ID, Name, Subject, Lecture Type, Status, Date, and Signature</li>
            <li>✓ All signatures are displayed in elegant Vladimir Script font for professional appearance</li>
            <li>✓ The first row is frozen for easy scrolling through records</li>
            <li>✓ Professional formatting with headers and borders</li>
          </ul>
        </div>
      </div>
    </Layout>
  )
}
