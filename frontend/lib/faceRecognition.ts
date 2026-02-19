// frontend Python Face Recognition API integration
// const PYTHON_API_URL = typeof window !== 'undefined' 
//   ? (process.env.NEXT_PUBLIC_FACE_API_URL || 'http://localhost:8000')
//   : 'http://localhost:8000'

const PYTHON_API_URL =
  process.env.NEXT_PUBLIC_FACE_API_URL || "https://aura-face-api.onrender.com";


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

    // Convert base64 images to File objects
    const files: File[] = []
    for (let i = 0; i < images.length; i++) {
      const base64 = images[i]
      // Remove data URI prefix if present
      const base64Data = base64.includes(',') ? base64.split(',')[1] : base64
      const binaryString = atob(base64Data)
      const bytes = new Uint8Array(binaryString.length)
      for (let j = 0; j < binaryString.length; j++) {
        bytes[j] = binaryString.charCodeAt(j)
      }
      const file = new File([bytes], `image_${i}.jpg`, { type: 'image/jpeg' })
      files.push(file)
    }

    // Create FormData with student_id and files
    const formData = new FormData()
    formData.append('student_id', studentId)
    for (const file of files) {
      formData.append('files', file)
    }

    // Use AbortController to enforce timeout (120 seconds for dlib)
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 120_000) // 2 minutes

    let response: Response
    try {
      response = await fetch(`${PYTHON_API_URL}/train`, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      })
    } catch (err) {
      if ((err as any)?.name === 'AbortError') {
        throw new Error('Training request timed out. Check Python API is running.')
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
    
    // Python API returns 'avg_embedding' - must be exactly 128-d
    if (!result.avg_embedding || !Array.isArray(result.avg_embedding)) {
      throw new Error('Invalid embedding received from Python API')
    }
    
    if (result.avg_embedding.length !== 128) {
      throw new Error(`Invalid embedding dimension: got ${result.avg_embedding.length}d, expected 128d`)
    }
    
    console.log(`✅ Valid embedding received: ${result.avg_embedding.length} dimensions`)
    return result.avg_embedding
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
    
    // Note: In the new architecture, students are stored in MongoDB on the Python API
    // No need to load them separately

    // Convert base64 to File object
    const base64Data = imageBase64.includes(',') ? imageBase64.split(',')[1] : imageBase64
    const binaryString = atob(base64Data)
    const bytes = new Uint8Array(binaryString.length)
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i)
    }
    const file = new File([bytes], 'image.jpg', { type: 'image/jpeg' })

    // Create FormData with 'file' field
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch(`${PYTHON_API_URL}/recognize`, {
      method: 'POST',
      body: formData
    })

    console.log('Python API response status:', response.status)

    if (!response.ok) {
      const errorText = await response.text()
      console.error('Python API error response:', errorText)
      throw new Error(`Recognition failed: HTTP ${response.status}`)
    }

    const result = await response.json()
    console.log('✅ Python API response:', result)
    
    // New API returns recognized boolean + student_id + confidence + distance
    if (result.recognized) {
      return [{
        name: result.student_id || 'Unknown',
        student_id: result.student_id,
        confidence: result.confidence,
        distance: result.distance,
        recognized: true
      }]
    }
    
    return []
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
