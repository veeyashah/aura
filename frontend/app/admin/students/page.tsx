"use client";

import { useEffect, useState } from "react";
import Layout from "@/components/layout/Layout";
import api from "@/lib/api";
import { toast } from "react-toastify";
import Link from "next/link";

export default function StudentsList() {
  const [students, setStudents] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedStudent, setSelectedStudent] = useState<any>(null);
  const [showSignatureModal, setShowSignatureModal] = useState(false);
  const [initials, setInitials] = useState("");
  const [signatureLoading, setSignatureLoading] = useState(false);

  // Always use Vladimir Script font (Excel built-in)
  const selectedFont = "Vladimir Script";

  useEffect(() => {
    fetchStudents();

    // Load Google Fonts
    loadGoogleFonts();

    // Refresh students list when page becomes visible (e.g., after returning from training)
    const handleVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        fetchStudents();
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);

    // Also refresh on focus (when user switches back to tab)
    window.addEventListener("focus", fetchStudents);

    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      window.removeEventListener("focus", fetchStudents);
    };
  }, []);

  const loadGoogleFonts = () => {
    // Vladimir Script is a system font, no need to load from Google Fonts
    // It's available in Excel and most systems by default
  };

  const fetchStudents = async () => {
    try {
      const response = await api.get("/admin/students");
      setStudents(response.data?.data || []);
    } catch (error: any) {
      toast.error(error.response?.data?.message || "Failed to load students");
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (studentId: string, studentName: string) => {
    if (
      !confirm(
        `Are you sure you want to delete ${studentName}? This will also delete all attendance records.`,
      )
    ) {
      return;
    }

    try {
      await api.delete(`/admin/students/${studentId}`);
      toast.success("Student deleted successfully");
      fetchStudents();
    } catch (error: any) {
      toast.error(error.response?.data?.message || "Failed to delete student");
    }
  };

  const openSignatureModal = (student: any) => {
    setSelectedStudent(student);
    setInitials("");

    // If student already has a signature, try to load it
    if (student.signature) {
      try {
        const signatureData = JSON.parse(student.signature);
        if (signatureData.initials) {
          setInitials(signatureData.initials);
        }
      } catch {
        // If it's not JSON (old canvas signature), just show empty form
      }
    }

    setShowSignatureModal(true);
  };

  const generateSignaturePreview = () => {
    if (!initials.trim()) return "";
    return initials.trim().toUpperCase();
  };

  const saveSignature = async () => {
    if (!selectedStudent) return;

    if (!initials.trim()) {
      toast.error("Please enter initials for the signature");
      return;
    }

    if (initials.trim().length > 5) {
      toast.error("Please enter maximum 5 characters for initials");
      return;
    }

    setSignatureLoading(true);
    try {
      // Save as JSON with initials and Vladimir Script font
      const signatureData = {
        initials: initials.trim().toUpperCase(),
        font: "Vladimir Script",
        timestamp: new Date().toISOString(),
        createdBy: "admin",
      };

      await api.put(`/admin/students/${selectedStudent._id}/signature`, {
        signature: JSON.stringify(signatureData),
      });

      toast.success("Signature saved successfully!");
      setShowSignatureModal(false);
      setSelectedStudent(null);
      setInitials("");
      fetchStudents();
    } catch (error: any) {
      toast.error(error.response?.data?.message || "Failed to save signature");
    } finally {
      setSignatureLoading(false);
    }
  };

  const resetSignature = () => {
    setInitials("");
  };

  return (
    <Layout role="admin">
      <div className="space-y-6">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 rounded-2xl p-8 shadow-2xl text-white">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-4xl font-bold mb-2">Students Management</h1>
              <p className="text-blue-100 text-lg">Manage all registered students</p>
            </div>
            <Link
              href="/admin/register"
              className="bg-white text-blue-600 px-6 py-3 rounded-xl hover:bg-blue-50 shadow-xl hover:shadow-2xl transition-all transform hover:-translate-y-0.5 font-bold flex items-center"
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              Register New Student
            </Link>
          </div>
        </div>

        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="inline-block animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-blue-600 mb-4"></div>
              <p className="text-xl text-gray-600">Loading students...</p>
            </div>
          </div>
        ) : (
          <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden">
            <div className="bg-gradient-to-r from-gray-50 to-gray-100 px-8 py-6 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold text-gray-800">All Students</h2>
                  <p className="text-gray-500 mt-1">{students.length} students registered</p>
                </div>
                <div className="flex gap-2">
                  <span className="bg-green-100 text-green-800 px-4 py-2 rounded-lg text-sm font-semibold border border-green-200">
                    ✅ {students.filter(s => s.isTrained).length} Trained
                  </span>
                  <span className="bg-red-100 text-red-800 px-4 py-2 rounded-lg text-sm font-semibold border border-red-200">
                    ❌ {students.filter(s => !s.isTrained).length} Untrained
                  </span>
                </div>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gradient-to-r from-slate-700 to-slate-800 text-white">
                  <tr>
                    <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider">
                      SAP ID
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider">
                      Roll Number
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider">
                      Name
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider">
                      Year
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider">
                      Batch
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider">
                      Department
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider">
                      Training
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider">
                      Signature
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {Array.isArray(students) && students.length > 0 ? (
                    students.map((student) => (
                      <tr key={student._id} className="hover:bg-blue-50 transition-colors duration-150">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="font-bold text-gray-900">{student.studentId}</span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="font-semibold text-gray-700">{student.rollNumber || "-"}</span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <div className="flex-shrink-0 h-10 w-10 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white font-bold">
                              {student.name?.charAt(0) || 'S'}
                            </div>
                            <div className="ml-4">
                              <div className="text-sm font-medium text-gray-900">{student.name}</div>
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-bold bg-blue-100 text-blue-800 border border-blue-200">
                            {student.year}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-bold bg-purple-100 text-purple-800 border border-purple-200">
                            {student.batch}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                          {student.department || "-"}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          {student.isTrained ? (
                            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-bold bg-green-100 text-green-800 border border-green-200">
                              ✅ Trained
                            </span>
                          ) : (
                            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-bold bg-red-100 text-red-800 border border-red-200">
                              ❌ Not Trained
                            </span>
                          )}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <button
                            onClick={() => openSignatureModal(student)}
                            className={`inline-flex items-center px-3 py-1 rounded-lg text-xs font-bold transition-all ${
                              student.signatureSubmitted
                                ? "bg-green-100 text-green-800 hover:bg-green-200 border border-green-200"
                                : "bg-orange-100 text-orange-800 hover:bg-orange-200 border border-orange-200"
                            }`}
                          >
                            {student.signatureSubmitted ? "✓ Signed" : "✗ Add Signature"}
                          </button>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex gap-2">
                            <Link
                              href={`/admin/students/${student._id}/training`}
                              className="inline-flex items-center px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-xs font-semibold transition-all shadow-sm hover:shadow-md"
                            >
                              <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                              </svg>
                              Train
                            </Link>
                            <button
                              onClick={() => handleDelete(student._id, student.name)}
                              className="inline-flex items-center px-3 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 text-xs font-semibold transition-all shadow-sm hover:shadow-md"
                            >
                              <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                              </svg>
                              Delete
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={9} className="px-6 py-16 text-center">
                        <div className="text-6xl mb-4">👥</div>
                        <p className="text-xl text-gray-500 mb-2">No students registered yet</p>
                        <p className="text-sm text-gray-400 mb-4">Start by registering your first student</p>
                        <Link
                          href="/admin/register"
                          className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl hover:from-blue-700 hover:to-purple-700 font-semibold shadow-lg"
                        >
                          <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                          </svg>
                          Register First Student
                        </Link>
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Adobe-Style Signature Modal */}
        {showSignatureModal && selectedStudent && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white p-8 rounded-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto shadow-2xl">
              <h2 className="text-2xl font-bold mb-2 text-gray-800">
                Digital Signature for {selectedStudent.name}
              </h2>
              <p className="text-gray-600 mb-6 text-sm">
                SAP ID: {selectedStudent.studentId} | Roll:{" "}
                {selectedStudent.rollNumber || "N/A"}
              </p>

              <div className="space-y-6">
                {/* Initials Input */}
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-3">
                    Student Initials *
                  </label>
                  <input
                    type="text"
                    value={initials}
                    onChange={(e) => setInitials(e.target.value.slice(0, 5))}
                    placeholder="e.g., VS, ABC, JD"
                    className="w-full px-6 py-4 border-2 border-gray-300 rounded-xl focus:ring-4 focus:ring-blue-200 focus:border-blue-500 text-center text-2xl font-semibold uppercase transition-all"
                    maxLength={5}
                    style={{ letterSpacing: "0.1em" }}
                  />
                  <p className="text-xs text-gray-500 mt-2 text-center">
                    2-5 characters
                  </p>
                </div>

                {/* Live Preview */}
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-3">
                    Signature Preview
                  </label>
                  <div className="border-2 border-dashed border-blue-300 rounded-xl p-12 bg-gradient-to-br from-blue-50 to-indigo-50 text-center">
                    {initials.trim() ? (
                      <div>
                        <div
                          className="text-7xl inline-block"
                          style={{
                            fontFamily: "Vladimir Script",
                            color: "#1e40af",
                          }}
                        >
                          {generateSignaturePreview()}
                        </div>
                        <p className="text-sm text-gray-600 mt-4">
                          Vladimir Script Font • Handwritten Style
                        </p>
                      </div>
                    ) : (
                      <div>
                        <div className="text-gray-400 text-lg mb-3">
                          Signature will appear here
                        </div>
                        <div
                          className="text-gray-300 text-5xl"
                          style={{ fontFamily: "Vladimir Script" }}
                        >
                          ABC
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex gap-4 mt-8 pt-6 border-t border-gray-200">
                <button
                  onClick={resetSignature}
                  className="flex-1 bg-gray-100 text-gray-700 px-6 py-3 rounded-xl hover:bg-gray-200 font-semibold transition-all shadow-sm"
                >
                  Clear
                </button>
                <button
                  onClick={saveSignature}
                  disabled={signatureLoading || !initials.trim()}
                  className="flex-1 bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-6 py-3 rounded-xl hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed font-semibold transition-all shadow-lg hover:shadow-xl"
                >
                  {signatureLoading ? "Saving..." : "✓ Save Signature"}
                </button>
                <button
                  onClick={() => {
                    setShowSignatureModal(false);
                    setSelectedStudent(null);
                    setInitials("");
                  }}
                  className="flex-1 bg-gray-300 text-gray-700 px-6 py-3 rounded-xl hover:bg-gray-400 font-semibold transition-all shadow-sm"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}
