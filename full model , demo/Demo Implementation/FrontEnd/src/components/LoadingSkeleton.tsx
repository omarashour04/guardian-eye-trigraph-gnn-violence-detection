type LoadingSkeletonProps = {
  className?: string
}

export default function LoadingSkeleton({ className = "" }: LoadingSkeletonProps) {
  return (
    <div
      className={`animate-pulse rounded-lg border border-[#c7e7f5] bg-[#f6fbff] ${className}`}
    >
      <div className="h-full min-h-24 rounded-lg bg-[linear-gradient(110deg,rgba(246,251,255,0.9),rgba(0,229,255,0.22),rgba(246,251,255,0.9))]" />
    </div>
  )
}
