'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'

export default function Splash() {
  const router = useRouter()
  const [isTransitioning, setIsTransitioning] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsTransitioning(true)
      setTimeout(() => {
        router.push('/login')
      }, 1000) // Wait for fade out animation
    }, 5000) // Show splash for 5 seconds

    return () => clearTimeout(timer)
  }, [router])

  return (
    <div className={`min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 transition-opacity duration-700 ${isTransitioning ? 'opacity-0' : 'opacity-100'}`}>
      <style>{`
        @keyframes logoFloat {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-20px); }
        }
        
        @keyframes gradientShift {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
        
        @keyframes borderGlow {
          0%, 100% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.3), 0 0 40px rgba(139, 92, 246, 0.2); }
          50% { box-shadow: 0 0 30px rgba(99, 102, 241, 0.6), 0 0 60px rgba(139, 92, 246, 0.4); }
        }
        
        @keyframes textGlow {
          0%, 100% { text-shadow: 0 0 10px rgba(99, 102, 241, 0.5), 0 0 20px rgba(139, 92, 246, 0.3); }
          50% { text-shadow: 0 0 20px rgba(99, 102, 241, 0.8), 0 0 40px rgba(139, 92, 246, 0.6); }
        }
        
        .logo-container {
          animation: logoFloat 3s ease-in-out infinite;
        }
        
        .logo-glow {
          animation: borderGlow 2s ease-in-out infinite;
        }
        
        .text-glow {
          animation: textGlow 2.5s ease-in-out infinite;
        }
      `}</style>

      <div className="text-center z-10">
        {/* Logo Container */}
        <div className="logo-container inline-block mb-8">
          <div className="logo-glow w-32 h-32 mx-auto bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 rounded-full flex items-center justify-center shadow-2xl">
            <div className="w-28 h-28 bg-slate-900 rounded-full flex items-center justify-center">
              <span className="text-5xl font-black bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                A
              </span>
            </div>
          </div>
        </div>

        {/* App Name */}
        <div className="space-y-4">
          <h1 className="text-glow text-6xl md:text-8xl font-black text-white tracking-tighter">
            AURA
          </h1>
          <p className="text-indigo-200 text-2xl md:text-3xl font-semibold mt-2">
            Automated Unified Recognition for Attendance
          </p>
          <p className="text-indigo-300 text-lg md:text-xl font-light tracking-wide animate-pulse mt-4">
            Smart Attendance Management
          </p>
        </div>

        {/* Loading Animation */}
        <div className="mt-12 flex justify-center gap-2">
          <div className="w-3 h-3 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '0s' }}></div>
          <div className="w-3 h-3 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
          <div className="w-3 h-3 bg-pink-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
        </div>
      </div>

      {/* Decorative Background Elements */}
      <div className="absolute top-20 left-10 w-64 h-64 bg-indigo-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob"></div>
      <div className="absolute top-40 right-10 w-64 h-64 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-2000"></div>
      <div className="absolute -bottom-8 left-20 w-64 h-64 bg-pink-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-4000"></div>
    </div>
  )
}
