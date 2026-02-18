'use client'

import { useEffect, useState } from 'react'
import Layout from '@/components/layout/Layout'
import api from '@/lib/api'
import { toast } from 'react-toastify'
import StatCard from '@/components/ui/StatCard'

export default function AdminDashboard() {
  const [stats, setStats] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchStats()
  }, [])

  const fetchStats = async () => {
    try {
      const response = await api.get('/admin/dashboard')
      setStats(response.data)
    } catch (error: any) {
      if (error.response?.status !== 401) {
        toast.error(error.response?.data?.message || 'Failed to load dashboard')
      }
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <Layout role="admin">
        <div className="flex items-center justify-center h-screen">
          <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-blue-600 mb-4"></div>
            <p className="text-xl text-gray-600">Loading dashboard...</p>
          </div>
        </div>
      </Layout>
    )
  }

  return (
    <Layout role="admin">
      <div className="space-y-8">
        {/* Header */}
        <div className="bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 rounded-2xl p-8 shadow-2xl text-white">
          <h1 className="text-4xl font-bold mb-2">Admin Dashboard</h1>
          <p className="text-purple-100 text-lg">System Overview & Analytics</p>
        </div>

        {/* Main Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <StatCard
            title="Total Students"
            value={stats?.totalStudents || 0}
            icon="👥"
            gradient="bg-gradient-to-br from-blue-500 via-blue-600 to-cyan-600"
            subtitle="Registered students"
          />
          <StatCard
            title="Trained Students"
            value={stats?.trainedStudents || 0}
            icon="✅"
            gradient="bg-gradient-to-br from-green-500 via-emerald-600 to-teal-600"
            subtitle="Face recognition ready"
          />
          <StatCard
            title="Total Faculty"
            value={stats?.totalFaculty || 0}
            icon="👨‍🏫"
            gradient="bg-gradient-to-br from-purple-500 via-purple-600 to-indigo-600"
            subtitle="Active faculty members"
          />
          <StatCard
            title="Today's Attendance"
            value={stats?.todayAttendance || 0}
            icon="📊"
            gradient="bg-gradient-to-br from-orange-500 via-red-500 to-pink-600"
            subtitle="Records marked today"
          />
        </div>

        {/* Secondary Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white rounded-2xl p-8 shadow-xl border-l-4 border-blue-500 hover:shadow-2xl transition-all duration-300">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-gray-800">Lecture Attendance</h3>
              <div className="text-3xl">📚</div>
            </div>
            <div className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent mb-2">
              {stats?.lectureAttendance || 0}
            </div>
            <p className="text-gray-500">Total lecture attendance records</p>
            <div className="mt-4 h-2 bg-gray-200 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full animate-pulse" style={{ width: '75%' }}></div>
            </div>
          </div>

          {/* Event Attendance removed from UI */}
        </div>

        {/* Recent Attendance Table */}
        <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden">
          <div className="bg-gradient-to-r from-gray-50 to-gray-100 px-8 py-6 border-b border-gray-200">
            <h2 className="text-2xl font-bold text-gray-800">Recent Attendance</h2>
            <p className="text-gray-500 mt-1">Latest attendance records across all classes</p>
          </div>
          
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">Student</th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">Subject/Event</th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">Faculty</th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">Date</th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">Type</th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">Status</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {stats?.recentAttendance?.length > 0 ? (
                  stats.recentAttendance.map((record: any, idx: number) => (
                    <tr key={idx} className="hover:bg-blue-50 transition-colors duration-150">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <div className="flex-shrink-0 h-10 w-10 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white font-bold">
                            {record.studentName?.charAt(0) || 'S'}
                          </div>
                          <div className="ml-4">
                            <div className="text-sm font-medium text-gray-900">{record.studentName}</div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700 font-medium">{record.subject}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">{record.facultyName}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                        {new Date(record.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-gray-100 text-gray-800 border border-gray-200`}>
                          📚 Lecture
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-bold ${
                          record.status === 'P' ? 'bg-green-100 text-green-800 border border-green-200' :
                          record.status === 'AG' ? 'bg-blue-100 text-blue-800 border border-blue-200' :
                          'bg-red-100 text-red-800 border border-red-200'
                        }`}>
                          {record.status === 'P' ? '✓ Present' : record.status === 'AG' ? '⚠ Absent (G)' : '✗ Absent'}
                        </span>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={6} className="px-6 py-16 text-center">
                      <div className="text-6xl mb-4">📋</div>
                      <p className="text-xl text-gray-500 mb-2">No recent attendance records</p>
                      <p className="text-sm text-gray-400">Attendance records will appear here once faculty starts marking</p>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <a
            href="/admin/students"
            className="group bg-gradient-to-br from-blue-500 to-cyan-600 rounded-xl p-6 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-1 text-white"
          >
            <div className="text-4xl mb-3">👥</div>
            <h3 className="text-lg font-bold mb-1">Manage Students</h3>
            <p className="text-blue-100 text-sm mb-3">View and manage all students</p>
            <div className="flex items-center text-sm font-semibold">
              <span>Go to Students</span>
              <svg className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </div>
          </a>

          <a
            href="/admin/register"
            className="group bg-gradient-to-br from-green-500 to-emerald-600 rounded-xl p-6 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-1 text-white"
          >
            <div className="text-4xl mb-3">➕</div>
            <h3 className="text-lg font-bold mb-1">Register Student</h3>
            <p className="text-green-100 text-sm mb-3">Add new student to system</p>
            <div className="flex items-center text-sm font-semibold">
              <span>Register Now</span>
              <svg className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </div>
          </a>

          <a
            href="/admin/faculty"
            className="group bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl p-6 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-1 text-white"
          >
            <div className="text-4xl mb-3">👨‍🏫</div>
            <h3 className="text-lg font-bold mb-1">Manage Faculty</h3>
            <p className="text-purple-100 text-sm mb-3">View and manage faculty</p>
            <div className="flex items-center text-sm font-semibold">
              <span>Go to Faculty</span>
              <svg className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </div>
          </a>

          <a
            href="/admin/export"
            className="group bg-gradient-to-br from-orange-500 to-red-600 rounded-xl p-6 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-1 text-white"
          >
            <div className="text-4xl mb-3">📥</div>
            <h3 className="text-lg font-bold mb-1">Export Data</h3>
            <p className="text-orange-100 text-sm mb-3">Download attendance reports</p>
            <div className="flex items-center text-sm font-semibold">
              <span>Export Now</span>
              <svg className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </div>
          </a>
        </div>
      </div>
    </Layout>
  )
}
