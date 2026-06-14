type StatusTone = "neutral" | "success" | "warning" | "danger"

type StatusCardProps = {
  label: string
  value: string
  description: string
  tone?: StatusTone
}

const toneStyles: Record<
  StatusTone,
  {
    border: string
    background: string
    label: string
    value: string
  }
> = {
  neutral: {
    border: "border-[#c7e7f5]",
    background: "bg-[#f6fbff]",
    label: "text-[#4b647f]",
    value: "text-[#041E42]",
  },
  success: {
    border: "border-emerald-500/20",
    background: "bg-emerald-50",
    label: "text-emerald-700",
    value: "text-emerald-700",
  },
  warning: {
    border: "border-amber-500/20",
    background: "bg-amber-500/10",
    label: "text-amber-300/80",
    value: "text-amber-200",
  },
  danger: {
    border: "border-red-500/20",
    background: "bg-red-500/10",
    label: "text-red-300/80",
    value: "text-red-200",
  },
}

export default function StatusCard({
  label,
  value,
  description,
  tone = "neutral",
}: StatusCardProps) {
  const styles = toneStyles[tone]

  return (
    <div
      className={`min-w-0 rounded-lg border p-4 ${styles.border} ${styles.background}`}
    >
      <p className={`text-xs font-medium uppercase ${styles.label}`}>
        {label}
      </p>
      <p className={`mt-2 break-words text-2xl font-bold ${styles.value}`}>
        {value}
      </p>
      <p className="mt-2 text-sm leading-6 text-[#4b647f]">{description}</p>
    </div>
  )
}
