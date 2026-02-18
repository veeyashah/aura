'use client'

import { useEffect, useState } from 'react'
import { useRouter, usePathname } from 'next/navigation'
import Sidebar from './Sidebar'

interface LayoutProps {
  children: React.ReactNode
  role: 'admin' | 'faculty' | 'student'
}

export default function Layout({ children, role }: LayoutProps) {
  const router = useRouter()
  const pathname = usePathname()
  const [isAuthorized, setIsAuthorized] = useState<boolean | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  useEffect(() => {
    const checkAuth = () => {
      const token = localStorage.getItem('token')
      const userRole = localStorage.getItem('userRole')

      if (!token || userRole !== role) {
        setIsAuthorized(false)
        // Only redirect if we're not already on the login page
        if (pathname !== '/') {
          router.push('/')
        }
      } else {
        setIsAuthorized(true)
      }
    }

    checkAuth()
  }, [role, router, pathname])

  // Show loading state while checking authentication
  if (isAuthorized === null) {
    return (
      <div className="flex min-h-screen bg-gray-100 items-center justify-center">
        <div className="text-xl">Loading...</div>
      </div>
    )
  }

  // Don't render if not authorized
  if (!isAuthorized) {
    return null
  }

  return (
    <div className="flex min-h-screen bg-gradient-to-br from-slate-50 to-gray-100">
      {/* Mobile Menu Button */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-slate-800 text-white rounded-lg shadow-lg"
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>

      {/* Mobile Overlay */}
      {sidebarOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black bg-opacity-50 z-40"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <Sidebar role={role} isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      
      <main className="flex-1 lg:p-8 p-4 pt-16 lg:pt-8 overflow-auto">
        {children}
      </main>
    </div>
  )
}
