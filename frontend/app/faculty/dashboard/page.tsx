'use client'

import { useEffect, useState } from 'react'
import Layout from '@/components/layout/Layout'
import api from '@/lib/api'
import { toast } from 'react-toastify'
import StatCard from '@/components/ui/StatCard'
import SubjectCard from '@/components/ui/SubjectCard'

export default function FacultyDashboard() {
  const [stats, setStats] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchStats()
  }, [])

  const fetchStats = async () => {
    try {
      const response = await api.get('/faculty/dashboard')
      setStats(response.data)
    } catch (error: any) {
      toast.error('Failed to load dashboard')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <Layout role="faculty">
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
    <Layout role="faculty">
      <div className="space-y-8">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 rounded-2xl p-8 shadow-2xl text-white">
          <h1 className="text-4xl font-bold mb-2">Faculty Dashboard</h1>
          <p className="text-blue-100 text-lg">Welcome back! Here's your overview</p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <StatCard
            title="Today's Activity"
            value={stats?.todayActivity || 0}
            icon="📊"
            gradient="bg-gradient-to-br from-blue-500 via-blue-600 to-indigo-600"
            subtitle="Classes marked today"
          />
          <StatCard
            title="Total Subjects"
            value={stats?.stats?.length || 0}
            icon="📚"
            gradient="bg-gradient-to-br from-purple-500 via-purple-600 to-pink-600"
            subtitle="Subjects assigned to you"
          />
        </div>

        {/* Subject-wise Attendance */}
        <div className="bg-white rounded-2xl p-8 shadow-xl border border-gray-100">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-2xl font-bold text-gray-800">Subject-wise Attendance</h2>
              <p className="text-gray-500 mt-1">Track attendance across all your subjects</p>
            </div>
            <div className="bg-gradient-to-r from-blue-500 to-purple-500 text-white px-4 py-2 rounded-lg text-sm font-semibold shadow-lg">
              {stats?.stats?.length || 0} Subjects
            </div>
          </div>

          {stats?.stats?.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {stats.stats.map((stat: any, idx: number) => (
                <SubjectCard
                  key={idx}
                  subject={stat.subject}
                  total={stat.total}
                  present={stat.present}
                  percentage={stat.percentage}
                />
              ))}
            </div>
          ) : (
            <div className="text-center py-16">
              <div className="text-6xl mb-4">📚</div>
              <p className="text-xl text-gray-500 mb-2">No subjects assigned yet</p>
              <p className="text-sm text-gray-400">Contact admin to assign subjects to your account</p>
            </div>
          )}
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <a
            href="/faculty/live-attendance"
            className="group bg-gradient-to-br from-green-500 to-emerald-600 rounded-xl p-6 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-1 text-white"
          >
            <div className="text-4xl mb-3">🎥</div>
            <h3 className="text-xl font-bold mb-2">Live Attendance</h3>
            <p className="text-green-100 text-sm">Start face recognition attendance</p>
            <div className="mt-4 flex items-center text-sm font-semibold">
              <span>Start Now</span>
              <svg className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </div>
          </a>

          <a
            href="/faculty/timetable"
            className="group bg-gradient-to-br from-orange-500 to-red-600 rounded-xl p-6 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-1 text-white"
          >
            <div className="text-4xl mb-3">📅</div>
            <h3 className="text-xl font-bold mb-2">Timetable</h3>
            <p className="text-orange-100 text-sm">View your class schedule</p>
            <div className="mt-4 flex items-center text-sm font-semibold">
              <span>View Schedule</span>
              <svg className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </div>
          </a>

          <a
            href="/faculty/export"
            className="group bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl p-6 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-1 text-white"
          >
            <div className="text-4xl mb-3">📥</div>
            <h3 className="text-xl font-bold mb-2">Export Data</h3>
            <p className="text-cyan-100 text-sm">Download attendance reports</p>
            <div className="mt-4 flex items-center text-sm font-semibold">
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
