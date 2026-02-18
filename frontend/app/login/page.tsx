'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Login from '@/components/auth/Login'

export default function LoginPage() {
  const router = useRouter()

  useEffect(() => {
    const token = localStorage.getItem('token')
    if (token) {
      const userRole = localStorage.getItem('userRole')
      if (userRole === 'admin') router.replace('/admin/dashboard')
      else if (userRole === 'faculty') router.replace('/faculty/dashboard')
    }
  }, [router])

  return <Login />
}
