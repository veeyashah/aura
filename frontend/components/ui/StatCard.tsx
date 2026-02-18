interface StatCardProps {
  title: string
  value: string | number
  icon: string
  gradient: string
  subtitle?: string
}

export default function StatCard({ title, value, icon, gradient, subtitle }: StatCardProps) {
  return (
    <div className={`relative overflow-hidden rounded-2xl p-6 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-1 ${gradient}`}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-white/80 mb-1">{title}</p>
          <h3 className="text-4xl font-bold text-white mb-1">{value}</h3>
          {subtitle && <p className="text-xs text-white/70">{subtitle}</p>}
        </div>
        <div className="text-5xl opacity-20">{icon}</div>
      </div>
      <div className="absolute bottom-0 right-0 w-32 h-32 bg-white/10 rounded-full -mr-16 -mb-16"></div>
    </div>
  )
}