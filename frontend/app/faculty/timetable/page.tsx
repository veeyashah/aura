'use client'

import { useEffect, useState } from 'react'
import Layout from '@/components/layout/Layout'
import api from '@/lib/api'
import { toast } from 'react-toastify'

export default function FacultyTimetable() {
  const [timetable, setTimetable] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchTimetable()
  }, [])

  const fetchTimetable = async () => {
    try {
      const response = await api.get('/faculty/timetable')
      // Ensure response.data is an array
      setTimetable(Array.isArray(response.data) ? response.data : [])
    } catch (error: any) {
      toast.error('Failed to load timetable')
      setTimetable([]) // Set empty array on error
    } finally {
      setLoading(false)
    }
  }

  const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

  return (
    <Layout role="faculty">
      <div className="space-y-6">
        {/* Header */}
        <div className="bg-gradient-to-r from-violet-600 via-purple-600 to-fuchsia-600 rounded-2xl p-8 shadow-2xl text-white">
          <h1 className="text-4xl font-bold mb-2">📅 My Timetable</h1>
          <p className="text-violet-100 text-lg">View your weekly class schedule</p>
        </div>

        {loading ? (
          <div className="flex justify-center items-center py-16">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-violet-600"></div>
          </div>
        ) : timetable.length === 0 ? (
          <div className="bg-white rounded-2xl shadow-xl border border-gray-100 p-12 text-center">
            <div className="text-6xl mb-4">📅</div>
            <h3 className="text-xl font-bold text-gray-800 mb-2">No Classes Scheduled</h3>
            <p className="text-gray-600">Your timetable is empty. Contact admin to add your classes.</p>
          </div>
        ) : (
          <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="min-w-full">
                <thead className="bg-gradient-to-r from-violet-700 to-purple-800 text-white">
                  <tr>
                    <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider sticky left-0 bg-violet-800">Time</th>
                    {days.map((day) => (
                      <th key={day} className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider">
                        {day}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {['09:00-10:00', '10:00-11:00', '11:00-12:00', '12:00-13:00', '14:00-15:00', '15:00-16:00'].map((timeSlot) => (
                    <tr key={timeSlot} className="hover:bg-gray-50 transition-colors">
                      <td className="px-6 py-4 whitespace-nowrap font-bold text-gray-800 bg-gradient-to-r from-gray-50 to-gray-100 sticky left-0">{timeSlot}</td>
                      {days.map((day) => {
                        const slot = Array.isArray(timetable) ? timetable.find(
                          (t) => t.day === day && t.timeSlot === timeSlot
                        ) : null
                        return (
                          <td key={day} className="px-6 py-4 min-w-[200px]">
                            {slot ? (
                              <div className="bg-gradient-to-br from-violet-50 to-purple-50 border-2 border-violet-200 rounded-xl p-4 hover:shadow-lg transition-all">
                                <div className="font-bold text-violet-900 mb-3 text-base">{slot.subject}</div>
                                <div className="flex flex-wrap gap-2 mb-2">
                                  <span className="text-xs bg-blue-200 text-blue-800 px-2 py-1 rounded-full font-bold">{slot.year}</span>
                                  <span className="text-xs bg-violet-200 text-violet-800 px-2 py-1 rounded-full font-semibold">{slot.lectureType}</span>
                                  <span className="text-xs bg-purple-200 text-purple-800 px-2 py-1 rounded-full font-semibold">{slot.batch}</span>
                                  <span className="text-xs bg-fuchsia-200 text-fuchsia-800 px-2 py-1 rounded-full font-semibold">Div {slot.division}</span>
                                </div>
                              </div>
                            ) : (
                              <div className="text-center text-gray-400 py-4">
                                <span className="text-2xl">−</span>
                              </div>
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
      </div>
    </Layout>
  )
}
