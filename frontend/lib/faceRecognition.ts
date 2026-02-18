// Python Face Recognition API integration
const PYTHON_API_URL = typeof window !== 'undefined' 
  ? (process.env.NEXT_PUBLIC_FACE_API_URL || 'http://localhost:8000')
  : 'http://localhost:8000'

// Helper: Compress image to smaller size - BALANCED quality/size
export const compressImageBase64 = (imageBase64: string, quality: number = 0.5): string => {
  if (imageBase64.length < 30000) return imageBase64 // Already small
  
  try {
    const img = new Image()
    img.src = imageBase64
    const canvas = document.createElement('canvas')
    canvas.width = img.width
    canvas.height = img.height
    const ctx = canvas.getContext('2d')
    if (!ctx) return imageBase64
    
    ctx.drawImage(img, 0, 0)
    return canvas.toDataURL('image/jpeg', quality) // Maintain quality for accuracy
  } catch (e) {
    return imageBase64 // Return original if compression fails
  }
}

export const loadModels = async () => {
  try {
    const response = await fetch(`${PYTHON_API_URL}/health`)
    if (response.ok) {
      const data = await response.json()
      console.log('✅ Python Face Recognition API connected:', data)
      return true
    }
    throw new Error(`Python API health check failed with status ${response.status}`)
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error)
    console.error('❌ Python Face Recognition API not available:', msg)
    throw new Error(`Python API unavailable: ${msg}`)
  }
}

export const loadStudentsToAPI = async (students: any[]) => {
  try {
    // Accept flexible embedding sizes: 24-d (OpenCV), 128-d (dlib), 512-d (DeepFace)
    const validSizes = [24, 128, 512]
    const validStudents = students.filter(s => 
      s.faceEmbeddings && 
      Array.isArray(s.faceEmbeddings) && 
      validSizes.includes(s.faceEmbeddings.length)
    )
    
    console.log(`📥 Loading ${validStudents.length}/${students.length} students to API (embedding sizes: ${validSizes.join(',')})`)
    
    if (validStudents.length === 0) {
      throw new Error(`No valid students found! Valid embedding sizes: ${validSizes.join(',')}`)
    }

    const response = await fetch(`${PYTHON_API_URL}/load-students`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ students: validStudents })
    })

    if (!response.ok) {
      const errorMsg = await response.text()
      console.error('Python API error:', errorMsg)
      throw new Error(`Python API error: ${response.status}`)
    }

    const result = await response.json()
    console.log(`✅ Python API loaded ${result.loaded_count}/${result.total_requested} students`)
    console.log(`   Skipped: ${result.skipped_count} (invalid embeddings)`)
    
    if (result.loaded_count === 0) {
      throw new Error(`Python API loaded 0 students - check embeddings are valid (${validSizes.join(',')}-d)`)
    }
    
    return result
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error)
    console.error('❌ Load students error:', msg)
    throw new Error(msg)
  }
}

export const trainStudent = async (studentId: string, images: string[]) => {
  try {
    if (images.length < 5) {
      throw new Error('Minimum 5 images required for training')
    }

    console.log(`🎓 Training student ${studentId} with ${images.length} images...`)

    // Quick health-check to provide a clearer error if Python API is down
    try {
      const healthResp = await fetch(`${PYTHON_API_URL}/health`, { method: 'GET' })
      if (!healthResp.ok) {
        throw new Error(`Python API health check failed: HTTP ${healthResp.status}`)
      }
    } catch (err) {
      const m = err instanceof Error ? err.message : String(err)
      throw new Error(`Python Face API unreachable (${PYTHON_API_URL}): ${m}`)
    }

    // Use AbortController to enforce a timeout for the training request
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 300_000) // 5 minutes (extended from 3)

    let response: Response
    try {
      // Compress images before sending to reduce payload
      const compressedImages = images.map(img => compressImageBase64(img, 0.3))
      
      response = await fetch(`${PYTHON_API_URL}/train`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ student_id: studentId, images: compressedImages }),
        signal: controller.signal
      })
    } catch (err) {
      if ((err as any)?.name === 'AbortError') {
        throw new Error('Training request timed out (5 minutes). Check Python API is running.')
      }
      const m = err instanceof Error ? err.message : String(err)
      throw new Error(`Failed to reach Python API: ${m}`)
    } finally {
      clearTimeout(timeout)
    }

    if (!response.ok) {
      try {
        const errorData = await response.json()
        const errorMsg = errorData?.detail || errorData?.message || `HTTP ${response.status}`
        throw new Error(`Python API error: ${errorMsg}`)
      } catch (parseError) {
        throw new Error(`Python API error: HTTP ${response.status} - ${response.statusText}`)
      }
    }

    let result
    try {
      result = await response.json()
    } catch (parseError) {
      throw new Error('Failed to parse Python API response')
    }

    console.log('✅ Training completed:', result)
    
    // Python API returns 'embedding' not 'encodings'
    if (!result.embedding || !Array.isArray(result.embedding)) {
      throw new Error('Invalid embedding received from Python API - expected 512-d array')
    }
    
    console.log(`✅ Valid embedding received: ${result.embedding.length} dimensions`)
    return result.embedding
  } catch (error) {
    // Ensure we always throw an Error with a string message
    const errorMessage = error instanceof Error ? error.message : String(error)
    console.error('❌ Training error:', errorMessage)
    throw new Error(errorMessage)
  }
}

export const recognizeFaces = async (imageBase64: string, students: any[] = []) => {
  try {
    console.log('🔍 Sending recognition request to Python API...')
    
    // Load students into API memory if provided
    if (students && students.length > 0) {
      await loadStudentsToAPI(students).catch(() => {
        // Silent fail if already loaded or if it fails
      })
    }

    const response = await fetch(`${PYTHON_API_URL}/recognize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image: imageBase64
      })
    })

    console.log('Python API response status:', response.status)

    if (!response.ok) {
      const errorText = await response.text()
      console.error('Python API error response:', errorText)
      throw new Error(`Recognition failed: HTTP ${response.status}`)
    }

    const result = await response.json()
    console.log('✅ Python API response:', result)
    return result.faces || []
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error)
    console.error('❌ Recognition error:', msg)
    throw new Error(msg)
  }
}

// Legacy functions for compatibility
export const detectFace = async (video: HTMLVideoElement) => {
  // This is now handled by the Python API
  return null
}

export const generateEmbedding = async (video: HTMLVideoElement) => {
  // This is now handled by the Python API
  return null
}

export const compareFaces = (embedding1: number[], embedding2: number[]): number => {
  // This is now handled by the Python API
  return 0
}

export const findBestMatch = (
  currentEmbedding: number[],
  storedEmbeddings: number[][],
  threshold: number = 0.6
) => {
  // This is now handled by the Python API
  return null
}
