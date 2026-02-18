'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useRouter } from 'next/navigation'
import { toast } from 'react-toastify'

interface SidebarProps {
  role: 'admin' | 'faculty' | 'student'
  isOpen?: boolean
  onClose?: () => void
}

export default function Sidebar({ role, isOpen = false, onClose }: SidebarProps) {
  const pathname = usePathname()
  const router = useRouter()

  const handleLogout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('userRole')
    localStorage.removeItem('userId')
    toast.success('Logged out successfully')
    router.push('/')
  }

  const handleLinkClick = () => {
    // Close sidebar on mobile when link is clicked
    if (onClose) {
      onClose()
    }
  }

  const adminLinks = [
    { href: '/admin/dashboard', label: 'Dashboard', icon: '📊' },
    { href: '/admin/students', label: 'Students', icon: '👥' },
    { href: '/admin/register', label: 'Register Student', icon: '➕' },
    { href: '/admin/faculty', label: 'Faculty', icon: '👨‍🏫' },
    { href: '/admin/timetable', label: 'Timetable', icon: '📅' },
    { href: '/admin/export', label: 'Export Attendance', icon: '📥' },
  ]

  const facultyLinks = [
    { href: '/faculty/dashboard', label: 'Dashboard', icon: '📊' },
    { href: '/faculty/live-attendance', label: 'Live Attendance', icon: '🎥' },
    { href: '/faculty/timetable', label: 'Timetable', icon: '📅' },
    { href: '/faculty/export', label: 'Export Attendance', icon: '📥' },
  ]

  const studentLinks = [
    { href: '/student/dashboard', label: 'Dashboard', icon: '📊' },
    { href: '/student/training', label: 'Face Training', icon: '📸' },
    { href: '/student/signature', label: 'Signature', icon: '✍️' },
    { href: '/student/attendance', label: 'My Attendance', icon: '📋' },
  ]

  const links = role === 'admin' ? adminLinks : role === 'faculty' ? facultyLinks : studentLinks

  return (
    <div className={`
      ${isOpen ? 'translate-x-0' : '-translate-x-full'} 
      lg:translate-x-0 
      fixed lg:static 
      inset-y-0 left-0 
      w-64 lg:w-64 
      bg-gradient-to-b from-slate-800 to-slate-900 
      text-white 
      min-h-screen 
      flex flex-col 
      shadow-xl 
      transition-transform duration-300 ease-in-out 
      z-50 lg:z-auto
    `}>
      {/* Close button for mobile */}
      <button
        onClick={onClose}
        className="lg:hidden absolute top-4 right-4 p-2 text-slate-300 hover:text-white"
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>

      <div className="p-6 border-b border-slate-700 bg-slate-800/50">
        <h2 className="text-xl lg:text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
          Attendance System
        </h2>
        <p className="text-sm text-slate-300 capitalize mt-2 font-medium">{role} Portal</p>
      </div>
      
      <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
        {links.map((link) => (
          <Link
            key={link.href}
            href={link.href}
            onClick={handleLinkClick}
            className={`flex items-center px-4 py-3 rounded-lg transition-all duration-200 ${
              pathname === link.href
                ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg shadow-blue-500/50 font-semibold'
                : 'text-slate-300 hover:bg-slate-700/50 hover:text-white'
            }`}
          >
            <span className="mr-3 text-lg">{link.icon}</span>
            <span className="text-sm lg:text-base">{link.label}</span>
          </Link>
        ))}
      </nav>
      
      <div className="p-4 border-t border-slate-700 bg-slate-800/50">
        <button
          onClick={handleLogout}
          className="w-full px-4 py-3 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 text-white rounded-lg transition-all duration-200 font-medium shadow-lg hover:shadow-red-500/50 text-sm lg:text-base"
        >
          Logout
        </button>
      </div>
    </div>
  )
}
