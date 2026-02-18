interface SubjectCardProps {
  subject: string
  total: number
  present: number
  percentage: string
}

export default function SubjectCard({ subject, total, present, percentage }: SubjectCardProps) {
  const percentNum = parseFloat(percentage)
  const getColor = () => {
    if (percentNum >= 75) return 'from-green-500 to-emerald-600'
    if (percentNum >= 60) return 'from-yellow-500 to-orange-500'
    return 'from-red-500 to-rose-600'
  }

  return (
    <div className="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 border border-gray-100 hover:border-blue-200 transform hover:-translate-y-1">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-gray-800 truncate">{subject}</h3>
        <span className={`text-2xl font-bold bg-gradient-to-r ${getColor()} bg-clip-text text-transparent`}>
          {percentage}%
        </span>
      </div>
      
      <div className="space-y-2 mb-4">
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">Total Classes</span>
          <span className="font-semibold text-gray-800">{total}</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">Present</span>
          <span className="font-semibold text-green-600">{present}</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">Absent</span>
          <span className="font-semibold text-red-600">{total - present}</span>
        </div>
      </div>

      <div className="relative w-full h-3 bg-gray-200 rounded-full overflow-hidden">
        <div
          className={`absolute top-0 left-0 h-full bg-gradient-to-r ${getColor()} rounded-full transition-all duration-500 shadow-lg`}
          style={{ width: `${Math.min(percentNum, 100)}%` }}
        >
          <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
        </div>
      </div>
    </div>
  )
}