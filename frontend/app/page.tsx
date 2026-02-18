'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Splash from '@/components/auth/Splash'

export default function Home() {
  const router = useRouter()

  useEffect(() => {
    const token = localStorage.getItem('token')
    if (token) {
      const userRole = localStorage.getItem('userRole')
      if (userRole === 'admin') router.replace('/admin/dashboard')
      else if (userRole === 'faculty') router.replace('/faculty/dashboard')
    }
  }, [router])

  return <Splash />
}
