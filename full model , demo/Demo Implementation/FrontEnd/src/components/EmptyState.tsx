import type { ReactNode } from "react"

type EmptyStateProps = {
  icon: ReactNode
  title: string
  description: string
}

export default function EmptyState({
  icon,
  title,
  description,
}: EmptyStateProps) {
  return (
    <div className="flex min-h-40 flex-col items-center justify-center rounded-lg border border-[#c7e7f5] bg-[#f6fbff] px-5 py-8 text-center">
      <div className="mb-4 flex size-12 items-center justify-center rounded-full border border-[#00A3E0]/25 bg-[#E6F7FF] text-[#0056D2]">
        {icon}
      </div>
      <h3 className="text-base font-semibold text-[#041E42]">{title}</h3>
      <p className="mt-2 max-w-sm text-sm leading-6 text-[#4b647f]">
        {description}
      </p>
    </div>
  )
}
