import {
  AlertTriangle,
  CheckCircle,
  Info,
  X,
  type LucideIcon,
} from "lucide-react"
import { useLanguage } from "@/context/LanguageContext"

type DemoNotificationType = "success" | "error" | "info" | "warning"

type DemoNotificationProps = {
  type: DemoNotificationType
  title: string
  message?: string
  onClose?: () => void
}

const notificationStyles: Record<
  DemoNotificationType,
  {
    icon: LucideIcon
    border: string
    background: string
    iconColor: string
    titleColor: string
  }
> = {
  success: {
    icon: CheckCircle,
    border: "border-emerald-500/25",
    background: "bg-white",
    iconColor: "text-emerald-600",
    titleColor: "text-emerald-700",
  },
  error: {
    icon: AlertTriangle,
    border: "border-red-500/25",
    background: "bg-white",
    iconColor: "text-red-600",
    titleColor: "text-red-700",
  },
  info: {
    icon: Info,
    border: "border-[#00A3E0]/25",
    background: "bg-white",
    iconColor: "text-[#0056D2]",
    titleColor: "text-[#041E42]",
  },
  warning: {
    icon: AlertTriangle,
    border: "border-amber-500/25",
    background: "bg-white",
    iconColor: "text-amber-600",
    titleColor: "text-amber-700",
  },
}

export default function DemoNotification({
  type,
  title,
  message,
  onClose,
}: DemoNotificationProps) {
  const { t, isArabic } = useLanguage()
  const styles = notificationStyles[type]
  const Icon = styles.icon

  return (
    <div className="fixed right-4 top-4 z-50 w-[calc(100%-2rem)] animate-in fade-in slide-in-from-top-2 duration-300 sm:right-6 sm:top-6 sm:w-full sm:max-w-sm">
      <div
        className={`rounded-lg border p-4 shadow-2xl shadow-[#0056D2]/15 backdrop-blur ${styles.border} ${styles.background}`}
      >
        <div className={`flex items-start gap-3 ${isArabic ? "text-right" : "text-left"}`}>
          <div className="mt-0.5 flex size-9 shrink-0 items-center justify-center rounded-lg border border-[#c7e7f5] bg-[#E6F7FF]">
            <Icon className={`size-5 ${styles.iconColor}`} />
          </div>

          <div className="min-w-0 flex-1">
            <p className={`text-sm font-semibold ${styles.titleColor}`}>
              {title}
            </p>
            {message && (
              <p className="mt-1 text-sm leading-6 text-[#4b647f]">
                {message}
              </p>
            )}
          </div>

          {onClose && (
            <button
              type="button"
              className="rounded-md p-1 text-[#4b647f] transition hover:bg-[#E6F7FF] hover:text-[#0056D2]"
              aria-label={t("closeNotification")}
              onClick={onClose}
            >
              <X className="size-4" />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
