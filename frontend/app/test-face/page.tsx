'use client'

import { useEffect, useState, useRef } from 'react'
import { loadModels, recognizeFaces } from '@/lib/faceRecognition'

export default function TestFace() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  
  const [apiConnected, setApiConnected] = useState(false)
  const [cameraStarted, setCameraStarted] = useState(false)
  const [lastDetection, setLastDetection] = useState<any>(null)
  const [faceDetected, setFaceDetected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [cameraDevices, setCameraDevices] = useState<MediaDeviceInfo[]>([])

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }
    }
  }, [])

  const checkCameraDevices = async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices()
      const cameras = devices.filter(device => device.kind === 'videoinput')
      setCameraDevices(cameras)
      console.log('Available cameras:', cameras)
    } catch (err: any) {
      setError(`Failed to enumerate devices: ${err.message}`)
    }
  }

  const connectToPythonAPI = async () => {
    try {
      setError(null)
      await loadModels()
      setApiConnected(true)
    } catch (err: any) {
      setError(`Failed to connect to Python API: ${err.message}`)
    }
  }

  const startCamera = async () => {
    try {
      setError(null)
      
      // Check if getUserMedia is available
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('getUserMedia is not supported in this browser')
      }

      console.log('Requesting camera access...')
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 640 }, 
          height: { ideal: 480 },
          facingMode: 'user'
        } 
      })
      
      console.log('Camera stream obtained:', stream)
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream
        
        // Wait for video to be ready
        videoRef.current.onloadedmetadata = () => {
          console.log('Video metadata loaded')
          // Wait a bit more for video to stabilize
          setTimeout(() => {
            setCameraStarted(true)
            console.log('Camera ready, dimensions:', videoRef.current?.videoWidth, 'x', videoRef.current?.videoHeight)
          }, 1000)
        }
        
        videoRef.current.onerror = (e) => {
          console.error('Video error:', e)
          setError('Video playback error')
        }
        
        // Also set camera started after a delay as backup
        setTimeout(() => {
          if (videoRef.current && videoRef.current.videoWidth > 0) {
            setCameraStarted(true)
          }
        }, 2000)
      }
    } catch (err: any) {
      console.error('Camera error:', err)
      let errorMessage = `Failed to access camera: ${err.message}`
      
      if (err.name === 'NotAllowedError') {
        errorMessage = 'Camera access denied. Please allow camera permissions and try again.'
      } else if (err.name === 'NotFoundError') {
        errorMessage = 'No camera found. Please connect a camera and try again.'
      } else if (err.name === 'NotReadableError') {
        errorMessage = 'Camera is already in use by another application.'
      } else if (err.name === 'OverconstrainedError') {
        errorMessage = 'Camera constraints cannot be satisfied.'
      }
      
      setError(errorMessage)
    }
  }

  const testFaceDetection = async () => {
    if (!videoRef.current || !apiConnected) {
      setError('Camera or Python API not ready')
      return
    }

    try {
      setError(null)
      console.log('Starting face detection test...')
      
      // Wait a moment for video to be fully ready
      await new Promise(resolve => setTimeout(resolve, 500))
      
      // Check if video is ready
      if (videoRef.current.readyState < 2) {
        setError('Video not ready. Please wait for camera to fully load and try again.')
        return
      }
      
      const videoWidth = videoRef.current.videoWidth
      const videoHeight = videoRef.current.videoHeight
      
      console.log('Video dimensions:', videoWidth, 'x', videoHeight)
      
      if (videoWidth === 0 || videoHeight === 0) {
        setError('Video dimensions are 0. Please wait for camera to initialize and try again.')
        return
      }
      
      // Capture current frame as base64
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      if (!ctx) {
        setError('Failed to create canvas context')
        return
      }

      canvas.width = videoWidth
      canvas.height = videoHeight
      
      // Draw the video frame to canvas
      ctx.drawImage(videoRef.current, 0, 0, videoWidth, videoHeight)
      
      // Convert to base64 with good quality
      const imageBase64 = canvas.toDataURL('image/jpeg', 0.9)
      
      console.log('Image captured, size:', imageBase64.length, 'characters')
      console.log('Sending to Python API...')

      // Send to Python API for face detection
      const faces = await recognizeFaces(imageBase64, [])
      
      console.log('Python API response:', faces)
      
      if (faces && faces.length > 0) {
        setLastDetection(faces)
        setFaceDetected(true)
        console.log('✅ Faces detected:', faces.length)
      } else {
        setFaceDetected(false)
        setLastDetection(null)
        console.log('ℹ️ No faces detected')
        // Don't set this as an error - it's normal if no face is visible
      }
    } catch (err: any) {
      console.error('❌ Face detection error:', err)
      setError(`Face detection error: ${err.message}`)
      setFaceDetected(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">Face Recognition Test</h1>
        
        <div className="bg-white p-6 rounded-lg shadow-md mb-6">
          <h2 className="text-xl font-semibold mb-4">Step 1: Connect to Python API</h2>
          <button
            onClick={connectToPythonAPI}
            disabled={apiConnected}
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {apiConnected ? '✅ Python API Connected' : 'Connect to Python API'}
          </button>
          <p className="text-sm text-gray-600 mt-2">
            Make sure the Python Face Recognition API is running on http://localhost:8000
          </p>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md mb-6">
          <h2 className="text-xl font-semibold mb-4">Step 2: Check Camera Devices</h2>
          <button
            onClick={checkCameraDevices}
            className="bg-yellow-600 text-white px-4 py-2 rounded hover:bg-yellow-700 mr-4"
          >
            Check Available Cameras
          </button>
          
          {cameraDevices.length > 0 && (
            <div className="mt-4">
              <h3 className="font-semibold">Available Cameras:</h3>
              <ul className="list-disc list-inside">
                {cameraDevices.map((device, index) => (
                  <li key={device.deviceId}>
                    {device.label || `Camera ${index + 1}`} ({device.deviceId.substring(0, 8)}...)
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md mb-6">
          <h2 className="text-xl font-semibold mb-4">Step 3: Start Camera</h2>
          <button
            onClick={startCamera}
            disabled={!apiConnected || cameraStarted}
            className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 disabled:opacity-50"
          >
            {cameraStarted ? '✅ Camera Started' : 'Start Camera'}
          </button>
          
          {cameraStarted && (
            <div className="mt-4">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                className="w-full max-w-md border-2 border-gray-300 rounded-lg"
              />
            </div>
          )}
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md mb-6">
          <h2 className="text-xl font-semibold mb-4">Step 4: Test Face Detection</h2>
          <button
            onClick={testFaceDetection}
            disabled={!cameraStarted || !apiConnected}
            className="bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700 disabled:opacity-50"
          >
            Test Face Detection
          </button>
          
          <div className="mt-4">
            <div className={`p-3 rounded ${faceDetected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
              {faceDetected ? `✅ ${lastDetection?.length || 0} Face(s) Detected!` : '❌ No Faces Detected'}
            </div>
            
            {lastDetection && lastDetection.length > 0 && (
              <div className="mt-2 p-3 bg-gray-100 rounded">
                <strong>Detection Results:</strong>
                {lastDetection.map((face: any, index: number) => (
                  <div key={index} className="mt-2 p-2 bg-white rounded border">
                    <div><strong>Face {index + 1}:</strong></div>
                    <div>Box: {face.box?.x}, {face.box?.y}, {face.box?.width}x{face.box?.height}</div>
                    <div>Confidence: {face.confidence ? (face.confidence * 100).toFixed(1) + '%' : 'N/A'}</div>
                    <div>Student: {face.student ? face.student.name : 'Unknown'}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
            <strong>Error:</strong> {error}
          </div>
        )}

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4">System Status</h2>
          <ul className="space-y-2">
            <li className={`flex items-center ${apiConnected ? 'text-green-600' : 'text-red-600'}`}>
              {apiConnected ? '✅' : '❌'} Python API: {apiConnected ? 'Connected' : 'Not Connected'}
            </li>
            <li className={`flex items-center ${cameraStarted ? 'text-green-600' : 'text-red-600'}`}>
              {cameraStarted ? '✅' : '❌'} Camera: {cameraStarted ? 'Active' : 'Inactive'}
            </li>
            <li className={`flex items-center ${faceDetected ? 'text-green-600' : 'text-gray-600'}`}>
              {faceDetected ? '✅' : '⏳'} Face Detection: {faceDetected ? 'Working' : 'Not Tested'}
            </li>
          </ul>
          
          <div className="mt-4 p-3 bg-blue-50 rounded">
            <h3 className="font-semibold text-blue-800">Instructions:</h3>
            <ol className="list-decimal list-inside text-sm text-blue-700 mt-2">
              <li>Start the Python API: <code>cd python-face-api && python main.py</code></li>
              <li>Click "Connect to Python API" to verify connection</li>
              <li>Check available cameras (optional)</li>
              <li>Start camera to begin video feed</li>
              <li>Test face detection to verify the system works</li>
            </ol>
          </div>
          
          {cameraStarted && videoRef.current && (
            <div className="mt-4 p-3 bg-gray-50 rounded">
              <h3 className="font-semibold text-gray-800">Video Debug Info:</h3>
              <div className="text-sm text-gray-700 mt-2">
                <div>Video Ready State: {videoRef.current.readyState}</div>
                <div>Video Dimensions: {videoRef.current.videoWidth} x {videoRef.current.videoHeight}</div>
                <div>Video Current Time: {videoRef.current.currentTime.toFixed(2)}s</div>
                <div>Video Paused: {videoRef.current.paused ? 'Yes' : 'No'}</div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}