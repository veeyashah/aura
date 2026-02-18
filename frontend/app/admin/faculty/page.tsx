'use client'

import { useEffect, useState } from 'react'
import Layout from '@/components/layout/Layout'
import api from '@/lib/api'
import { toast } from 'react-toastify'

export default function FacultyManagement() {
  const [faculty, setFaculty] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [showAddModal, setShowAddModal] = useState(false)
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    facultyName: '',
    subjects: [] as string[],
  })
  const [newSubject, setNewSubject] = useState('')

  useEffect(() => {
    fetchFaculty()
  }, [])

  const fetchFaculty = async () => {
    try {
      const response = await api.get('/admin/faculty')
      setFaculty(response.data)
    } catch (error: any) {
      toast.error(error.response?.data?.message || 'Failed to load faculty')
    } finally {
      setLoading(false)
    }
  }

  const handleAddSubject = () => {
    if (newSubject.trim() && !formData.subjects.includes(newSubject.trim())) {
      setFormData({
        ...formData,
        subjects: [...formData.subjects, newSubject.trim()],
      })
      setNewSubject('')
    }
  }

  const handleRemoveSubject = (subject: string) => {
    setFormData({
      ...formData,
      subjects: formData.subjects.filter(s => s !== subject),
    })
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      await api.post('/admin/faculty', formData)
      toast.success('Faculty added successfully!')
      setShowAddModal(false)
      setFormData({ email: '', password: '', facultyName: '', subjects: [] })
      fetchFaculty()
    } catch (error: any) {
      toast.error(error.response?.data?.message || 'Failed to add faculty')
    }
  }

  const handleUpdate = async (facultyId: string, updates: any) => {
    try {
      await api.put(`/admin/faculty/${facultyId}`, updates)
      toast.success('Faculty updated successfully!')
      fetchFaculty()
    } catch (error: any) {
      toast.error(error.response?.data?.message || 'Failed to update faculty')
    }
  }

  const handleDelete = async (facultyId: string, facultyName: string) => {
    if (!confirm(`Are you sure you want to delete ${facultyName}?`)) {
      return
    }
    try {
      await api.delete(`/admin/faculty/${facultyId}`)
      toast.success('Faculty deleted successfully')
      fetchFaculty()
    } catch (error: any) {
      toast.error(error.response?.data?.message || 'Failed to delete faculty')
    }
  }

  return (
    <Layout role="admin">
      <div className="space-y-6">
        {/* Header */}
        <div className="bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 rounded-2xl p-8 shadow-2xl text-white">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-4xl font-bold mb-2">👨‍🏫 Faculty Management</h1>
              <p className="text-indigo-100 text-lg">Manage faculty members and their subjects</p>
            </div>
            <button
              onClick={() => setShowAddModal(true)}
              className="bg-white text-indigo-600 px-6 py-3 rounded-xl hover:bg-indigo-50 shadow-xl hover:shadow-2xl transition-all transform hover:-translate-y-0.5 font-bold"
            >
              + Add Faculty
            </button>
          </div>
        </div>

        {loading ? (
          <div className="flex justify-center items-center py-16">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
          </div>
        ) : faculty.length === 0 ? (
          <div className="bg-white rounded-2xl shadow-xl border border-gray-100 p-12 text-center">
            <div className="text-6xl mb-4">👨‍🏫</div>
            <h3 className="text-xl font-bold text-gray-800 mb-2">No Faculty Members Yet</h3>
            <p className="text-gray-600 mb-6">Get started by adding your first faculty member</p>
            <button
              onClick={() => setShowAddModal(true)}
              className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white px-6 py-3 rounded-xl hover:from-indigo-700 hover:to-purple-700 shadow-xl transition-all"
            >
              + Add Faculty
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {faculty.map((fac) => (
              <div key={fac._id} className="bg-white rounded-2xl shadow-xl border border-gray-100 p-6 hover:shadow-2xl transition-all transform hover:-translate-y-1">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center">
                    <div className="w-12 h-12 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex items-center justify-center text-white font-bold text-xl mr-3">
                      {fac.facultyName.charAt(0).toUpperCase()}
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-gray-800">{fac.facultyName}</h3>
                      <p className="text-gray-500 text-sm">{fac.email}</p>
                    </div>
                  </div>
                </div>
                <div className="mb-4">
                  <p className="text-sm font-bold text-gray-700 mb-2">📚 Subjects:</p>
                  <div className="flex flex-wrap gap-2">
                    {fac.subjects?.length > 0 ? (
                      fac.subjects.map((subject: string, idx: number) => (
                        <span key={idx} className="bg-gradient-to-r from-blue-100 to-indigo-100 text-indigo-800 px-3 py-1 rounded-full text-xs font-semibold border border-indigo-200">
                          {subject}
                        </span>
                      ))
                    ) : (
                      <span className="text-gray-400 text-xs">No subjects assigned</span>
                    )}
                  </div>
                </div>
                <div className="flex gap-2 pt-4 border-t border-gray-200">
                  <button
                    onClick={() => handleDelete(fac._id, fac.facultyName)}
                    className="flex-1 bg-gradient-to-r from-red-500 to-rose-600 text-white px-4 py-2 rounded-xl hover:from-red-600 hover:to-rose-700 text-sm font-semibold shadow-lg hover:shadow-xl transition-all"
                  >
                    🗑️ Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Add Faculty Modal */}
        {showAddModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full">
              <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-t-2xl p-6 text-white">
                <h2 className="text-2xl font-bold">Add Faculty Member</h2>
                <p className="text-indigo-100 text-sm mt-1">Fill in the details below</p>
              </div>
              <form onSubmit={handleSubmit} className="p-6 space-y-4">
                <div>
                  <label className="block text-sm font-bold text-gray-700 mb-2">Faculty Name *</label>
                  <input
                    type="text"
                    required
                    value={formData.facultyName}
                    onChange={(e) => setFormData({ ...formData, facultyName: e.target.value })}
                    className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all"
                    placeholder="Enter full name"
                  />
                </div>
                <div>
                  <label className="block text-sm font-bold text-gray-700 mb-2">Email *</label>
                  <input
                    type="email"
                    required
                    value={formData.email}
                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                    className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all"
                    placeholder="faculty@example.com"
                  />
                </div>
                <div>
                  <label className="block text-sm font-bold text-gray-700 mb-2">Password *</label>
                  <input
                    type="password"
                    required
                    value={formData.password}
                    onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                    className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all"
                    placeholder="Enter password"
                  />
                </div>
                <div>
                  <label className="block text-sm font-bold text-gray-700 mb-2">Subjects</label>
                  <div className="flex gap-2 mb-3">
                    <input
                      type="text"
                      value={newSubject}
                      onChange={(e) => setNewSubject(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), handleAddSubject())}
                      placeholder="Add subject"
                      className="flex-1 px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all"
                    />
                    <button
                      type="button"
                      onClick={handleAddSubject}
                      className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white px-6 py-3 rounded-xl hover:from-indigo-700 hover:to-purple-700 font-semibold shadow-lg transition-all"
                    >
                      Add
                    </button>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {formData.subjects.map((subject, idx) => (
                      <span
                        key={idx}
                        className="bg-gradient-to-r from-blue-100 to-indigo-100 text-indigo-800 px-3 py-2 rounded-full text-sm font-semibold flex items-center gap-2 border border-indigo-200"
                      >
                        {subject}
                        <button
                          type="button"
                          onClick={() => handleRemoveSubject(subject)}
                          className="text-indigo-600 hover:text-indigo-800 font-bold"
                        >
                          ×
                        </button>
                      </span>
                    ))}
                  </div>
                </div>
                <div className="flex gap-3 pt-4">
                  <button
                    type="submit"
                    className="flex-1 bg-gradient-to-r from-indigo-600 to-purple-600 text-white py-3 rounded-xl hover:from-indigo-700 hover:to-purple-700 font-bold shadow-xl hover:shadow-2xl transition-all"
                  >
                    Add Faculty
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setShowAddModal(false)
                      setFormData({ email: '', password: '', facultyName: '', subjects: [] })
                    }}
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
