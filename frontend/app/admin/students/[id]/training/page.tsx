'use client'

import { useEffect, useState, useRef } from 'react'
import { useParams, useRouter } from 'next/navigation'
import Layout from '@/components/layout/Layout'
import api from '@/lib/api'
import { toast } from 'react-toastify'
import { loadModels, trainStudent } from '@/lib/faceRecognition'

export default function StudentTraining() {
  const params = useParams()
  const router = useRouter()
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  
  const [student, setStudent] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [training, setTraining] = useState(false)
  const [processing, setProcessing] = useState(false) // New: tracks AI processing phase
  const [progress, setProgress] = useState(0)
  const [capturedCount, setCapturedCount] = useState(0)
  const [targetImages] = useState(20) // Keep 20 images for HIGH ACCURACY training
  const [capturedImages, setCapturedImages] = useState<string[]>([])
  const [cameraReady, setCameraReady] = useState(false)
  const [messageIndex, setMessageIndex] = useState(0) // For cycling messages

  // Engaging messages during processing
  const processingMessages = [
    '🔍 Analyzing your facial features...',
    '🧠 Training the AI model...',
    '📊 Creating your biometric profile...',
    '✨ Building facial recognition model...',
    '⚡ Processing biometric data...',
    '🎯 Calibrating accuracy metrics...',
    '🔐 Securing your face recognition...',
    '🎓 Your face model is being trained...'
  ]

  useEffect(() => {
    fetchStudent()
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }
    }
  }, [])

  // Cycle through processing messages every 2 seconds
  useEffect(() => {
    if (!processing) return
    
    const interval = setInterval(() => {
      setMessageIndex(prev => (prev + 1) % processingMessages.length)
    }, 2000)
    
    return () => clearInterval(interval)
  }, [processing, processingMessages.length])

  const fetchStudent = async () => {
    try {
      const response = await api.get(`/admin/students/${params.id}`)
      // Backend returns { success: true, data: { student data } }
      const studentData = response.data.data || response.data
      console.log('📚 Loaded student:', studentData)
      setStudent(studentData)
    } catch (error: any) {
      toast.error('Failed to load student data')
      router.push('/admin/students')
    } finally {
      setLoading(false)
    }
  }

  const startCamera = async () => {
    try {
      await loadModels() // Check Python API connection
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 640 }, 
          height: { ideal: 480 }
        } 
      })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream
        setCameraReady(false)
        // Wait for video to actually start playing before capturing
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play()
          toast.info('Camera warming up...')
          // Give camera 1 second to warm up
          setTimeout(() => {
            setCameraReady(true)
            startTraining()
          }, 1000)
        }
      }
    } catch (error: any) {
      const errorMsg = error instanceof Error ? error.message : String(error)
      if (errorMsg.includes('Python')) {
        toast.error('Python Face Recognition API is not running. Please start the Python server first.')
      } else if (errorMsg.includes('camera') || errorMsg.includes('getUserMedia')) {
        toast.error('Failed to access camera. Check permissions.')
      } else {
        toast.error(errorMsg || 'Failed to start camera')
      }
    }
  }

  const captureImageAsBase64 = (): string | null => {
    if (!videoRef.current) return null
    
    try {
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      if (!ctx) return null
      
      canvas.width = videoRef.current.videoWidth
      canvas.height = videoRef.current.videoHeight
      ctx.drawImage(videoRef.current, 0, 0)
      
      return canvas.toDataURL('image/jpeg', 0.5)
    } catch (error) {
      console.error('Capture error:', error)
      return null
    }
  }

  const startTraining = async () => {
    if (!videoRef.current || !videoRef.current.srcObject) {
      toast.error('Please start camera first')
      return
    }

    setTraining(true)
    setCapturedCount(0)
    setProgress(0)
    const images: string[] = []
    const captureTargetImages = 10 // 10 images in 1 second = ultra-fast training
    const totalTime = 1000
    const interval = totalTime / captureTargetImages

    toast.info('Capturing... Keep face steady!')

    for (let i = 0; i < captureTargetImages; i++) {
      const imageBase64 = captureImageAsBase64()
      if (imageBase64) {
        images.push(imageBase64)
        setCapturedImages([...images])
      }
      
      const currentCount = i + 1
      setCapturedCount(currentCount)
      setProgress((currentCount / captureTargetImages) * 100)
      
      if (i < captureTargetImages - 1) {
        await new Promise(resolve => setTimeout(resolve, interval))
      }
    }

    if (images.length < 5) {
      toast.error(`Not enough images captured (${images.length}/${captureTargetImages}). Please try again.`)
      setTraining(false)
      return
    }

    // Start processing phase - show loading modal
    setProcessing(true)
    setMessageIndex(0)

    try {
      console.log('📤 Sending images to Python API for training...')
      
      // Validate student and studentId exist
      if (!student || !student.studentId) {
        throw new Error('Student information not loaded. Please refresh the page.')
      }
      
      console.log(`📚 Training student: ${student.studentId}`)
      const embeddings = await trainStudent(student.studentId, images)
      
      console.log(`✅ Received embeddings: ${embeddings.length} dimensions`)
      console.log(`   - Embedding sample: [${embeddings.slice(0, 5).map((n: number) => n.toFixed(3)).join(', ')}...]`)
      
      console.log('📤 Posting embeddings to backend training endpoint...')
      const response = await api.post(`/admin/students/${params.id}/training`, {
        faceEmbeddings: embeddings,
      })
      
      console.log('✅ Backend response:', response.data)
      
      if (response.data.success) {
        console.log('✅ Training successful!')
        setProcessing(false)
        toast.success(`✨ Training completed! ${response.data.data.embeddingSize} dimensions saved.`)
        await new Promise(resolve => setTimeout(resolve, 500))
        router.push('/admin/students')
      } else {
        const errorMsg = response.data.message || 'Unknown error'
        console.error('❌ Training response not successful:', errorMsg)
        setProcessing(false)
        toast.error(`Training failed: ${errorMsg}`)
        setTraining(false)
      }
    } catch (error: any) {
      setProcessing(false)
      console.error('❌ Training error caught:', error)
      
      // Ensure we have a string message
      let errorMessage = 'Training failed'
      
      // Handle different error types
      if (error instanceof Error) {
        // Standard Error object
        errorMessage = error.message
      } else if (typeof error === 'object' && error !== null) {
        // Try to extract message from object
        if (error.response?.data?.message) {
          errorMessage = String(error.response.data.message)
        } else if (error.response?.data?.detail) {
          errorMessage = String(error.response.data.detail)
        } else if (error.message) {
          errorMessage = String(error.message)
        } else if (error.detail) {
          errorMessage = String(error.detail)
        } else {
          // Fallback: stringify the parts we can get
          errorMessage = `Error: ${JSON.stringify(error).substring(0, 100)}`
        }
      } else if (typeof error === 'string') {
        errorMessage = error
      }
      
      // Log detailed error info
      console.error(`Error source: ${errorMessage}`)
      console.error(`Full error object:`, error)
      
      if (error.response) {
        console.error(`Response status: ${error.response.status}`)
        console.error(`Response data:`, error.response.data)
      }
      
      // Show appropriate toast message
      if (errorMessage.includes('Python')) {
        toast.error('Python API error. Check if server is running on port 8000.')
      } else {
        toast.error(errorMessage)
      }
      
      setTraining(false)
    }
  }

  if (loading) {
    return (
      <Layout role="admin">
        <div className="text-center py-8">Loading...</div>
      </Layout>
    )
  }

  return (
    <Layout role="admin">
      <div>
        <h1 className="text-3xl font-bold mb-6">
          Train Student: {student?.name} ({student?.studentId})
        </h1>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <div className="mb-4">
            <button
              onClick={startCamera}
              disabled={training}
              className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-6 py-3 rounded-lg hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 shadow-lg"
            >
              {training ? 'Training in Progress...' : 'Start Camera & Auto-Capture'}
            </button>
            <p className="text-sm text-gray-600 mt-2">
              Camera will start and automatically capture {targetImages} images in ~1 second
            </p>
          </div>

          <div className="mb-4">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="w-full max-w-md border-2 border-gray-300 rounded-lg shadow-lg"
            />
          </div>

          {training && (
            <div className="mb-4 p-6 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border-2 border-blue-300 shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <div className="text-2xl font-bold text-blue-800">
                  Capturing Images: <span className="text-purple-600">{capturedCount}</span> / <span className="text-gray-600">{targetImages}</span>
                </div>
                <div className="text-xl font-semibold text-blue-700">
                  {Math.round(progress)}%
                </div>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-8 mb-3 shadow-inner">
                <div
                  className="bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 h-8 rounded-full transition-all duration-200 ease-out flex items-center justify-end pr-2"
                  style={{ width: `${progress}%` }}
                >
                  {progress > 10 && (
                    <span className="text-white text-xs font-semibold">{capturedCount}/{targetImages}</span>
                  )}
                </div>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-blue-700 font-medium">
                  {capturedCount === 50 ? '✅ Capture Complete! Processing...' : '⏳ Capturing...'}
                </span>
                <span className="text-gray-600">
                  {capturedImages.length > 0 && `${capturedImages.length} images captured`}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Processing/Loading Modal */}
        {processing && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 backdrop-blur-sm">
            <div className="bg-white rounded-3xl shadow-2xl p-12 max-w-md w-full mx-4 transform transition-all">
              {/* Circular Loading Spinner */}
              <div className="flex justify-center mb-8">
                <div className="relative w-32 h-32">
                  {/* Outer rotating circle */}
                  <svg className="absolute inset-0 w-32 h-32 transform -rotate-90 animate-spin" style={{ animationDuration: '3s' }} viewBox="0 0 100 100">
                    <circle cx="50" cy="50" r="45" fill="none" stroke="#3B82F6" strokeWidth="3" strokeDasharray="141 188" opacity="0.8" />
                  </svg>
                  
                  {/* Inner pulsing circle */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full animate-pulse shadow-lg"></div>
                  </div>

                  {/* Center icon */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-3xl">🧠</span>
                  </div>
                </div>
              </div>

              {/* Engaging Message */}
              <div className="text-center mb-6">
                <h2 className="text-2xl font-bold text-gray-800 mb-4">
                  Training in Progress...
                </h2>
                <p className="text-lg text-purple-600 font-medium h-8 transition-all duration-500">
                  {processingMessages[messageIndex]}
                </p>
              </div>

              {/* Progress Dots */}
              <div className="flex justify-center gap-2 mb-6">
                {[0, 1, 2].map((i) => (
                  <div
                    key={i}
                    className="w-3 h-3 rounded-full bg-blue-400 animate-bounce"
                    style={{ animationDelay: `${i * 0.2}s` }}
                  />
                ))}
              </div>

              {/* Motivational Text */}
              <p className="text-center text-gray-600 text-sm">
                Sit back and relax. Your facial recognition model is being trained with advanced AI...
              </p>
            </div>
          </div>
        )}
      </div>
    </Layout>
  )
}
