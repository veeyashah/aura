"use client";

import { useEffect, useState, useRef } from "react";
import Layout from "@/components/layout/Layout";
import api from "@/lib/api";
import { toast } from "react-toastify";

interface DetectedFace {
  name: string;
  student_id: string;
  distance: number; // Changed from confidence
  box: [number, number, number, number];
  recognized: boolean;
}

interface Student {
  studentId: string;
  name: string;
  rollNumber: string;
  year: string;
  batch: string;
  faceEmbeddings?: number[];
  isTrained: boolean;
}

export default function LiveAttendance() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const [subjects, setSubjects] = useState<string[]>([]);
  const [selectedSubject, setSelectedSubject] = useState("");
  const [lectureType, setLectureType] = useState("Lecture");
  const [year, setYear] = useState("FY");
  const [batch, setBatch] = useState("ALL");
  const [division] = useState("A");
  const [students, setStudents] = useState<Student[]>([]);

  const [isLiveActive, setIsLiveActive] = useState(false);
  const [detectedFaces, setDetectedFaces] = useState<DetectedFace[]>([]);
  const [markedStudents, setMarkedStudents] = useState<Set<string>>(new Set());
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [processing, setProcessing] = useState(false);
  const [cameraStatus, setCameraStatus] = useState<string>("Not started");

  // Confidence threshold system
  const CONFIDENCE_THRESHOLDS = {
    AUTO_MARK: 0.22, // distance ≤ 0.22 (similarity ≥ 0.78)
    MANUAL_CONFIRM: 0.35, // distance ≤ 0.35 (similarity ≥ 0.65)
    MANUAL_REVIEW: Infinity, // distance > 0.35 (similarity < 0.65)
  };
  const [pendingConfirmation, setPendingConfirmation] =
    useState<DetectedFace | null>(null);
  const [thresholdStats, setThresholdStats] = useState({
    autoMarked: 0,
    manualConfirm: 0,
    manualReview: 0,
  });

  useEffect(() => {
    fetchSubjects();
    return () => {
      if (stream) stream.getTracks().forEach((track) => track.stop());
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  useEffect(() => {
    if (selectedSubject) fetchStudents();
  }, [selectedSubject, year, batch, lectureType]);

  // Handle frame processing when live becomes active
  useEffect(() => {
    if (isLiveActive) {
      console.log("🚀 Live active → starting frame processing");
      startFrameProcessing();
    } else {
      // Stop frame processing when live becomes inactive
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }
  }, [isLiveActive]);

  const fetchSubjects = async () => {
    try {
      const response = await api.get("/faculty/subjects");
      setSubjects(response.data);
      if (response.data.length > 0) setSelectedSubject(response.data[0]);
    } catch (error: any) {
      toast.error("Failed to load subjects");
    }
  };

  const fetchStudents = async () => {
    try {
      const params = new URLSearchParams();
      params.append("year", year);
      if (batch !== "ALL") params.append("batch", batch);

      console.log(`Fetching students: year=${year}, batch=${batch}`);
      const response = await api.get(`/faculty/students?${params.toString()}`);
      console.log(`Received ${response.data.length} students from API`);

      const trainedStudents = response.data.filter((s: any) => s.isTrained);
      console.log(`Filtered to ${trainedStudents.length} trained students`);

      setStudents(trainedStudents);
      checkTodayAttendance(trainedStudents);
    } catch (error: any) {
      console.error("Failed to load students:", error);
      toast.error(
        `Failed to load students: ${error.response?.data?.message || error.message}`,
      );
    }
  };

  const checkTodayAttendance = async (studentList: Student[]) => {
    try {
      const today = new Date().toISOString().split("T")[0];
      const response = await api.get(
        `/faculty/attendance?subject=${selectedSubject}&startDate=${today}&endDate=${today}`,
      );
      const markedToday = new Set<string>();
      response.data.forEach((record: any) => {
        if (record.status === "P") markedToday.add(record.studentId);
      });
      setMarkedStudents(markedToday);
    } catch (error) {
      console.log("No attendance records found for today");
    }
  };

  const startLiveRecognition = async () => {
    try {
      if (lectureType === "Practical" && !batch) {
        toast.error("Please select a batch (B1 or B2) for practical sessions");
        return;
      }

      if (students.length === 0) {
        toast.error("No trained students found");
        return;
      }

      setProcessing(true);
      setCameraStatus("Starting camera...");

      // FIRST: Enable video element rendering by setting isLiveActive = true
      // This ensures videoRef is attached to the DOM
      setIsLiveActive(true);

      // Small delay to ensure DOM update
      await new Promise((resolve) => setTimeout(resolve, 100));

      // Step 1: Get camera stream
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      });

      console.log("✅ Camera stream obtained");
      setStream(mediaStream);

      // NOW videoRef should exist
      if (!videoRef.current) {
        throw new Error("Video element not ready - please try again");
      }

      videoRef.current.srcObject = mediaStream;

      // Wait for video to actually load and be ready to play
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(
          () => reject(new Error("Video load timeout")),
          5000,
        );

        const onMetadata = () => {
          clearTimeout(timeout);
          console.log(
            `✅ Video loaded: ${videoRef.current!.videoWidth}x${videoRef.current!.videoHeight}`,
          );
          videoRef
            .current!.play()
            .catch((e) => console.error("Play error:", e));
          videoRef.current!.removeEventListener("loadedmetadata", onMetadata);
          videoRef.current!.removeEventListener("error", onError);
          resolve(null);
        };

        const onError = () => {
          clearTimeout(timeout);
          videoRef.current!.removeEventListener("loadedmetadata", onMetadata);
          videoRef.current!.removeEventListener("error", onError);
          reject(new Error("Video failed to load"));
        };

        videoRef.current!.addEventListener("loadedmetadata", onMetadata);
        videoRef.current!.addEventListener("error", onError);
      });

      setCameraStatus("Loading students to Python API...");

      // Step 2: CRITICAL - Load students to Python API FIRST
      // Accept flexible embedding sizes: 24-d (OpenCV new), 128-d (dlib legacy), 512-d (future DeepFace)
      const validSizes = [24, 128, 512];
      const studentsForAPI = students
        .filter(
          (s) =>
            s.faceEmbeddings &&
            Array.isArray(s.faceEmbeddings) &&
            validSizes.includes(s.faceEmbeddings.length),
        )
        .map((s) => ({
          studentId: s.studentId,
          name: s.name,
          faceEmbeddings: s.faceEmbeddings,
        }));

      console.log(
        `📥 Loading ${studentsForAPI.length} students to Python API (accepting ${validSizes.join("/")}d embeddings)`,
      );

      if (studentsForAPI.length === 0) {
        const embeddingSizes =
          students
            .filter((s) => Array.isArray(s.faceEmbeddings))
            .map((s) => `${s.name}: ${s.faceEmbeddings!.length}d`)
            .join(", ") || "No embeddings found";
        throw new Error(
          `No valid embeddings found. Found: ${embeddingSizes} (valid: ${validSizes.join("/")}d)`,
        );
      }

      const apiUrl = process.env.NEXT_PUBLIC_FACE_API_URL || "http://localhost:8000";

      const loadResponse = await fetch(`${apiUrl}/load-students`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ students: studentsForAPI }),
      });

      if (!loadResponse.ok) {
        const errorText = await loadResponse.text();
        console.error(
          `❌ Python API error: ${loadResponse.status} - ${errorText}`,
        );
        throw new Error(
          `Failed to load students to Python API: ${loadResponse.status} ${loadResponse.statusText}`,
        );
      }

      const loadResult = await loadResponse.json();
      console.log(
        `✅ Loaded ${loadResult.loaded_count} students to Python API (skipped ${loadResult.skipped_count})`,
      );

      if (loadResult.loaded_count === 0) {
        throw new Error(
          "Python API failed to load any students - check embeddings format",
        );
      }

      // Step 3: Frame processing will start automatically via useEffect
      setCameraStatus("🎥 Camera active - waiting for faces");
      setProcessing(false);

      // startFrameProcessing triggered by useEffect when isLiveActive changes
      toast.success(
        `✅ Live recognition started with ${loadResult.loaded_count} students`,
      );
    } catch (error: any) {
      console.error("Camera error:", error);
      setCameraStatus("Error: " + error.message);
      toast.error("Failed to start: " + error.message);
      setProcessing(false);
      setIsLiveActive(false);
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        setStream(null);
      }
    }
  };

  const startFrameProcessing = () => {
    console.log("🔥 Frame processing started");

    intervalRef.current = setInterval(async () => {
      if (!isLiveActive || !videoRef.current || !canvasRef.current) {
        console.debug(
          "⏸️ Skipping frame: isLiveActive=",
          isLiveActive,
          "videoRef=",
          !!videoRef.current,
          "canvasRef=",
          !!canvasRef.current,
        );
        return;
      }

      try {
        const video = videoRef.current;

        // Skip if video not ready - check both dimensions AND readyState
        if (
          video.videoWidth === 0 ||
          video.videoHeight === 0 ||
          video.readyState < video.HAVE_CURRENT_DATA
        ) {
          console.debug(
            `⏳ Video not ready (w:${video.videoWidth}, h:${video.videoHeight}, state:${video.readyState})`,
          );
          return;
        }

        console.log("📸 Processing frame...");

        const tempCanvas = document.createElement("canvas");
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        const tempCtx = tempCanvas.getContext("2d");
        if (!tempCtx) {
          console.error("❌ Failed to get canvas context");
          return;
        }

        tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
        // Use 0.65 quality for recognition (original working)
        const imageData = tempCanvas.toDataURL("image/jpeg", 0.65);

        const apiUrl = process.env.NEXT_PUBLIC_FACE_API_URL || "http://localhost:8000";

        console.log(`🔍 Sending frame to ${apiUrl}/recognize...`);

        // Send ONLY image - students are already loaded in Python API memory
        const response = await fetch(`${apiUrl}/recognize`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: imageData }),
        });

        console.log(`✅ Python API responded: ${response.status}`);

        if (!response.ok) {
          console.error(
            `❌ Python API error: ${response.status} ${response.statusText}`,
          );
          const errorText = await response.text();
          console.error("Error details:", errorText);
          return;
        }

        const result = await response.json();
        console.log("📊 Recognition result:", result);

        if (result.success && result.faces) {
          console.log(`👤 Found ${result.faces.length} face(s)`);
          handleRecognitionResult(result.faces);
        } else {
          console.debug("ℹ️ No faces detected or recognition failed:", result);
        }
      } catch (error) {
        console.error("❌ Frame processing error:", error);

        if (error instanceof Error) {
          console.error("Error stack:", error.stack);
        }
      }
    }, 1000); // Process every 1 second

    console.log("✅ Frame processing interval set (every 1s)");
  };

  const handleRecognitionResult = (faces: DetectedFace[]) => {
    setDetectedFaces(faces);

    // Classify faces by confidence threshold
    let autoMarked = 0,
      manualConfirm = 0,
      manualReview = 0;
    faces.forEach((face: DetectedFace) => {
      if (face.recognized) {
        if (face.distance <= CONFIDENCE_THRESHOLDS.AUTO_MARK) autoMarked++;
        else if (face.distance <= CONFIDENCE_THRESHOLDS.MANUAL_CONFIRM)
          manualConfirm++;
        else manualReview++;
      }
    });
    setThresholdStats({ autoMarked, manualConfirm, manualReview });

    faces.forEach((face: DetectedFace) => {
      // Only auto-mark high confidence; medium confidence requires teacher confirmation
      if (
        face.recognized &&
        face.distance <= CONFIDENCE_THRESHOLDS.AUTO_MARK &&
        !Array.from(markedStudents).includes(face.student_id)
      ) {
        markAttendance(
          face.student_id,
          face.name,
          face.distance,
          "auto-marked",
        );
      }
    });
    drawFaceBoxes(faces);
  };

  const drawFaceBoxes = (faces: DetectedFace[]) => {
    if (!canvasRef.current || !videoRef.current) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    faces.forEach((face: DetectedFace) => {
      const [left, top, right, bottom] = face.box;
      const width = right - left;
      const height = bottom - top;

      // Color based on CONFIDENCE THRESHOLD (Distance Metric)
      let color = "#ff0000"; // Red = unknown
      let thresholdLabel = "";
      if (face.recognized) {
        if (Array.from(markedStudents).includes(face.student_id)) {
          color = "#0066ff"; // Blue = already marked
          thresholdLabel = " [MARKED]";
        } else if (face.distance <= CONFIDENCE_THRESHOLDS.AUTO_MARK) {
          color = "#00ff00"; // Green = HIGH confidence (≤0.22) → AUTO-MARK
          thresholdLabel = " [AUTO ✅]";
        } else if (face.distance <= CONFIDENCE_THRESHOLDS.MANUAL_CONFIRM) {
          color = "#ffaa00"; // Orange = MEDIUM confidence (0.22-0.35) → CONFIRM
          thresholdLabel = " [CONFIRM ⚠️]";
        } else {
          color = "#ff3333"; // Red = LOW confidence (>0.35) → REVIEW
          thresholdLabel = " [REVIEW ❌]";
        }
      } else {
        // Draw boxes even for unrecognized faces to verify detection
        color = "#ff0000"; // Red for unknown faces
        thresholdLabel = " [Unknown]";
      }

      ctx.strokeStyle = color;
      ctx.lineWidth = 4;
      ctx.strokeRect(left, top, width, height);

      const text = face.recognized ? face.name + thresholdLabel : "Unknown";
      const distanceText = `d: ${face.distance.toFixed(3)}`; // Show distance

      ctx.font = "bold 20px Arial";
      const textWidth = Math.max(
        ctx.measureText(text).width,
        ctx.measureText(distanceText).width,
      );
      const textX = right + 10;
      const textY = top;

      ctx.fillStyle = color;
      ctx.fillRect(textX, textY, textWidth + 20, 65);
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 16px Arial";
      ctx.fillText(text, textX + 10, textY + 28);
      ctx.font = "bold 14px Arial";
      ctx.fillText(distanceText, textX + 10, textY + 55);
    });
  };

  const markAttendance = async (
    studentId: string,
    studentName: string,
    distance: number,
    thresholdType: string = "manual-review",
  ) => {
    try {
      // Capture the current frame for storage - LOW QUALITY for speed
      if (videoRef.current && canvasRef.current) {
        const tempCanvas = document.createElement("canvas");
        tempCanvas.width = videoRef.current.videoWidth;
        tempCanvas.height = videoRef.current.videoHeight;
        const ctx = tempCanvas.getContext("2d");
        if (ctx) {
          ctx.drawImage(videoRef.current, 0, 0);
          // Use 0.65 quality (original working)
          const imageBase64 = tempCanvas.toDataURL("image/jpeg", 0.65);

          // Save face capture to backend (async - don't wait)
          api
            .post("/attendance/capture-face", {
              studentId,
              studentName,
              imageBase64,
              recognized: true,
              confidence: distance,
              subject: selectedSubject,
              facultyName: "System",
              year,
              division,
            })
            .catch((err) => console.log("Face save async:", err.message));
        }
      }

      // Mark attendance immediately (don't wait for face save)
      const response = await api.post("/faculty/attendance", {
        studentId,
        studentName,
        subject: selectedSubject,
        status: "P",
        lectureType,
        year,
        batch: batch === "ALL" ? "ALL" : batch,
        division,
        confidence: distance,
        confidenceThreshold: thresholdType,
      });

      if (response.data.alreadyMarked) {
        setMarkedStudents((prev) => new Set([...Array.from(prev), studentId]));
        return;
      }

      setMarkedStudents((prev) => new Set([...Array.from(prev), studentId]));
      const thresholdLabel =
        thresholdType === "auto-marked" ? "✅ AUTO" : "⚠️ MANUAL";
      toast.success(
        `${thresholdLabel} ${studentName} marked (d: ${distance.toFixed(3)})`,
      );
    } catch (error: any) {
      console.error("Mark attendance error:", error);
      toast.error(
        `Failed to mark ${studentName}: ${error.response?.data?.message || error.message}`,
      );
    }
  };

  const manualMarkAttendance = async (student: Student, status: "P" | "A") => {
    try {
      // Capture face if available for present status - LOW QUALITY
      if (status === "P" && videoRef.current && canvasRef.current) {
        const tempCanvas = document.createElement("canvas");
        tempCanvas.width = videoRef.current.videoWidth;
        tempCanvas.height = videoRef.current.videoHeight;
        const ctx = tempCanvas.getContext("2d");
        if (ctx) {
          ctx.drawImage(videoRef.current, 0, 0);
          // 0.65 quality (original working)
          const imageBase64 = tempCanvas.toDataURL("image/jpeg", 0.65);

          // Async - don't wait for face save
          api
            .post("/attendance/capture-face", {
              studentId: student.studentId,
              studentName: student.name,
              imageBase64,
              recognized: true,
              confidence: 1.0,
              subject: selectedSubject,
              facultyName: "System",
              year,
              division,
            })
            .catch((err) => console.log("Face save async:", err.message));
        }
      }

      // Mark attendance immediately
      await api.post("/faculty/attendance", {
        studentId: student.studentId,
        studentName: student.name,
        subject: selectedSubject,
        status,
        lectureType,
        year,
        batch: batch === "ALL" ? "ALL" : batch,
        division,
        confidence: 1.0,
      });
      if (status === "P") {
        setMarkedStudents(
          (prev) => new Set([...Array.from(prev), student.studentId]),
        );
        toast.success(`✅ ${student.name} marked present manually`);
      } else {
        setMarkedStudents((prev) => {
          const newSet = new Set(Array.from(prev));
          newSet.delete(student.studentId);
          return newSet;
        });
        toast.info(`${student.name} marked absent`);
      }
    } catch (error: any) {
      toast.error(
        `Failed to mark ${student.name}: ${error.response?.data?.message || error.message}`,
      );
    }
  };

  const stopLiveRecognition = async () => {
    setIsLiveActive(false);
    setProcessing(true);
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
    if (videoRef.current) videoRef.current.srcObject = null;

    const presentStudentIds = Array.from(markedStudents);
    const absentStudents = students.filter(
      (student) => !presentStudentIds.includes(student.studentId),
    );

    if (absentStudents.length > 0) {
      try {
        for (const student of absentStudents) {
          await api.post("/faculty/attendance", {
            studentId: student.studentId,
            studentName: student.name,
            subject: selectedSubject,
            status: "A",
            lectureType,
            year,
            batch: batch === "ALL" ? "ALL" : batch,
            division,
            confidence: 0.0,
          });
        }
        const attendancePercentage = (
          (markedStudents.size / students.length) *
          100
        ).toFixed(1);
        toast.success(
          `✅ Attendance completed! ${markedStudents.size} present, ${absentStudents.length} absent (${attendancePercentage}%)`,
        );
      } catch (error: any) {
        toast.error("Failed to mark some students as absent");
      }
    }
    setDetectedFaces([]);
    setProcessing(false);
  };

  const attendancePercentage =
    students.length > 0
      ? ((markedStudents.size / students.length) * 100).toFixed(1)
      : "0";

  return (
    <Layout role="faculty">
      <div className="space-y-6">
        {/* Header */}
        <div className="bg-gradient-to-r from-green-600 via-emerald-600 to-teal-600 rounded-2xl p-8 shadow-2xl text-white">
          <h1 className="text-4xl font-bold mb-2">
            🎥 Live Face Recognition Attendance
          </h1>
          <p className="text-green-100 text-lg">
            Real-time attendance marking with AI-powered face recognition
          </p>
        </div>

        {/* Class Settings */}
        <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden">
          <div className="bg-gradient-to-r from-gray-50 to-gray-100 px-8 py-6 border-b border-gray-200">
            <h2 className="text-2xl font-bold text-gray-800">
              Class Configuration
            </h2>
            <p className="text-gray-500 mt-1">
              Select class details before starting attendance
            </p>
          </div>
          <div className="p-8">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
              <div>
                <label className="block text-sm font-bold text-gray-700 mb-2">
                  Subject
                </label>
                <select
                  value={selectedSubject}
                  onChange={(e) => setSelectedSubject(e.target.value)}
                  className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-green-500 bg-white transition-all"
                  disabled={isLiveActive}
                >
                  {subjects.map((subject) => (
                    <option key={subject} value={subject}>
                      {subject}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-bold text-gray-700 mb-2">
                  Year
                </label>
                <select
                  value={year}
                  onChange={(e) => setYear(e.target.value)}
                  className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-green-500 bg-white transition-all"
                  disabled={isLiveActive}
                >
                  <option value="FY">First Year (FY)</option>
                  <option value="SY">Second Year (SY)</option>
                  <option value="TY">Third Year (TY)</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-bold text-gray-700 mb-2">
                  Batch
                </label>
                <select
                  value={batch}
                  onChange={(e) => setBatch(e.target.value)}
                  className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-green-500 bg-white transition-all"
                  disabled={isLiveActive}
                >
                  {lectureType === "Lecture" ? (
                    <option value="ALL">All Batches</option>
                  ) : (
                    <>
                      <option value="">Select Batch</option>
                      <option value="B1">Batch 1 (B1)</option>
                      <option value="B2">Batch 2 (B2)</option>
                    </>
                  )}
                </select>
              </div>
              <div>
                <label className="block text-sm font-bold text-gray-700 mb-2">
                  Type
                </label>
                <select
                  value={lectureType}
                  onChange={(e) => {
                    setLectureType(e.target.value);
                    if (e.target.value === "Lecture") setBatch("ALL");
                    else if (e.target.value === "Practical") setBatch("");
                  }}
                  className="w-full px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-green-500 bg-white transition-all"
                  disabled={isLiveActive}
                >
                  <option value="Lecture">Lecture</option>
                  <option value="Practical">Practical</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Live Dashboard */}
        <div className="bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 rounded-2xl p-8 shadow-xl border-2 border-blue-200">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h2 className="text-2xl font-bold text-gray-800">
                Live Attendance Dashboard
              </h2>
              <p className="text-gray-600 mt-1">
                Real-time face recognition and marking
              </p>
            </div>
            <div className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              {attendancePercentage}%
            </div>
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <div className="bg-white rounded-xl p-6 text-center shadow-lg border-2 border-gray-200 hover:shadow-xl transition-all">
              <div className="text-4xl font-bold text-gray-800">
                {students.length}
              </div>
              <div className="text-sm text-gray-600 font-semibold mt-1">
                Total Students
              </div>
            </div>
            <div className="bg-gradient-to-br from-green-500 to-emerald-600 rounded-xl p-6 text-center shadow-lg hover:shadow-xl transition-all text-white">
              <div className="text-4xl font-bold">{markedStudents.size}</div>
              <div className="text-sm font-semibold mt-1 text-green-100">
                Present
              </div>
            </div>
            <div className="bg-gradient-to-br from-red-500 to-rose-600 rounded-xl p-6 text-center shadow-lg hover:shadow-xl transition-all text-white">
              <div className="text-4xl font-bold">
                {students.length - markedStudents.size}
              </div>
              <div className="text-sm font-semibold mt-1 text-red-100">
                Absent
              </div>
            </div>
            <div className="bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl p-6 text-center shadow-lg hover:shadow-xl transition-all text-white">
              <div className="text-4xl font-bold">{detectedFaces.length}</div>
              <div className="text-sm font-semibold mt-1 text-blue-100">
                Faces Detected
              </div>
            </div>
          </div>
          <div className="text-center">
            {!isLiveActive ? (
              <button
                onClick={startLiveRecognition}
                disabled={processing || students.length === 0}
                className="bg-gradient-to-r from-green-600 to-emerald-600 text-white px-12 py-5 rounded-2xl hover:from-green-700 hover:to-emerald-700 disabled:opacity-50 font-bold text-xl shadow-2xl hover:shadow-3xl transition-all transform hover:-translate-y-1 disabled:transform-none"
              >
                {processing ? (
                  <span className="flex items-center justify-center">
                    <svg
                      className="animate-spin -ml-1 mr-3 h-6 w-6 text-white"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>
                    Starting...
                  </span>
                ) : (
                  <span className="flex items-center justify-center">
                    <svg
                      className="w-6 h-6 mr-2"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                      />
                    </svg>
                    🎥 Start Live Recognition
                  </span>
                )}
              </button>
            ) : (
              <button
                onClick={stopLiveRecognition}
                disabled={processing}
                className="bg-gradient-to-r from-red-600 to-rose-600 text-white px-12 py-5 rounded-2xl hover:from-red-700 hover:to-rose-700 font-bold text-xl shadow-2xl hover:shadow-3xl transition-all transform hover:-translate-y-1 disabled:opacity-50"
              >
                {processing
                  ? "⏳ Finishing..."
                  : "⏹️ Stop & Complete Attendance"}
              </button>
            )}
          </div>
          {students.length === 0 && (
            <div className="bg-yellow-50 border-2 border-yellow-300 rounded-xl p-6 mt-6">
              <div className="flex items-center">
                <svg
                  className="w-6 h-6 text-yellow-600 mr-3"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                  />
                </svg>
                <p className="text-yellow-800 font-semibold">
                  ⚠️ No trained students found. Please ensure students have
                  completed face training.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Live Camera Feed */}
        {isLiveActive && (
          <div className="bg-white p-6 rounded-xl shadow-lg mb-6 mx-2 lg:mx-0 border border-gray-200">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold text-gray-800">
                📹 Live Camera Feed
              </h2>
              <div className="text-sm font-medium text-blue-600 bg-blue-50 px-3 py-1 rounded-lg">
                {cameraStatus}
              </div>
            </div>
            <div
              style={{
                position: "relative",
                width: "100%",
                paddingBottom: "66.67%",
                background: "#000",
                borderRadius: "12px",
                overflow: "hidden",
              }}
            >
              <video
                ref={videoRef}
                autoPlay={true}
                playsInline={true}
                muted={true}
                controls={false}
                onLoadedMetadata={() => {
                  console.log(
                    `✅ Video metadata ready: ${videoRef.current?.videoWidth}x${videoRef.current?.videoHeight}`,
                  );
                }}
                onError={(e) => {
                  console.error("❌ Video error:", e);
                  toast.error("Camera stream error");
                }}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  height: "100%",
                  objectFit: "cover",
                  backgroundColor: "#000",
                }}
              />
              <canvas
                ref={canvasRef}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  height: "100%",
                  pointerEvents: "none",
                  zIndex: 10,
                }}
              />
            </div>
            <div className="mt-4 bg-gray-50 rounded-xl p-4 border border-gray-200">
              <div className="text-sm font-semibold text-gray-800 mb-3">
                Detection Legend:
              </div>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
                <div className="flex items-center bg-white p-2 rounded-lg shadow-sm">
                  <div className="w-4 h-4 bg-green-500 rounded mr-2"></div>
                  <span className="font-medium">Excellent (d ≤ 0.35)</span>
                </div>
                <div className="flex items-center bg-white p-2 rounded-lg shadow-sm">
                  <div className="w-4 h-4 bg-yellow-500 rounded mr-2"></div>
                  <span className="font-medium">Good (d ≤ 0.50)</span>
                </div>
                <div className="flex items-center bg-white p-2 rounded-lg shadow-sm">
                  <div className="w-4 h-4 bg-red-500 rounded mr-2"></div>
                  <span className="font-medium">Unknown (d &gt; 0.50)</span>
                </div>
                <div className="flex items-center bg-white p-2 rounded-lg shadow-sm">
                  <div className="w-4 h-4 bg-blue-500 rounded mr-2"></div>
                  <span className="font-medium">Already Marked</span>
                </div>
              </div>
              <div className="mt-3 text-xs text-gray-600 bg-white p-2 rounded-lg">
                🔄 Processing every 1s • dlib CNN • Distance threshold: 0.50 •
                Lower = better match
              </div>
            </div>
          </div>
        )}

        {/* Student List with Manual Marking */}
        <div className="bg-white p-6 rounded-xl shadow-lg mb-6 mx-2 lg:mx-0 border border-gray-200">
          <h2 className="text-xl font-semibold mb-4 text-gray-800">
            📋 Student List & Manual Marking
          </h2>
          {students.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50 border-b-2 border-gray-200">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">
                      Roll No
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">
                      Name
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">
                      Year
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">
                      Batch
                    </th>
                    <th className="px-4 py-3 text-center text-xs font-semibold text-gray-700 uppercase">
                      Status
                    </th>
                    <th className="px-4 py-3 text-center text-xs font-semibold text-gray-700 uppercase">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {students.map((student) => {
                    const isPresent = Array.from(markedStudents).includes(
                      student.studentId,
                    );
                    return (
                      <tr
                        key={student.studentId}
                        className={`hover:bg-gray-50 transition-colors ${isPresent ? "bg-green-50" : ""}`}
                      >
                        <td className="px-4 py-3 text-sm font-medium text-gray-900">
                          {student.rollNumber || "N/A"}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-900">
                          {student.name}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-600">
                          {student.year}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-600">
                          {student.batch}
                        </td>
                        <td className="px-4 py-3 text-center">
                          {isPresent ? (
                            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-green-100 text-green-800 border border-green-300">
                              ✓ Present
                            </span>
                          ) : (
                            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-gray-100 text-gray-600 border border-gray-300">
                              − Absent
                            </span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-center">
                          <div className="flex gap-2 justify-center">
                            {!isPresent ? (
                              <button
                                onClick={() =>
                                  manualMarkAttendance(student, "P")
                                }
                                disabled={isLiveActive}
                                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 text-xs font-semibold shadow-sm hover:shadow-md transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                              >
                                ✓ Mark Present
                              </button>
                            ) : (
                              <button
                                onClick={() =>
                                  manualMarkAttendance(student, "A")
                                }
                                disabled={isLiveActive}
                                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 text-xs font-semibold shadow-sm hover:shadow-md transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                              >
                                ✗ Mark Absent
                              </button>
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              No students found. Please select class configuration above.
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}
