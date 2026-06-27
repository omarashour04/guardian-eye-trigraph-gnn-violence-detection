import { useEffect, useState } from "react"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useLanguage } from "@/context/LanguageContext"

type VerdictPanelProps = {
  verdict: "violence" | "non-violence"
  confidence: number
}

export default function VerdictPanel({
  verdict,
  confidence,
}: VerdictPanelProps) {
  const { t, isArabic } = useLanguage()
  const isViolence = verdict === "violence"
  const modelConfidence = isViolence ? confidence : 1 - confidence
  const confidencePercent = Math.round(Math.min(1, Math.max(0, modelConfidence)) * 100)
  const [displayConfidence, setDisplayConfidence] = useState(0)
  const [isTransitioning, setIsTransitioning] = useState(false)

  useEffect(() => {
    let animationFrame = 0
    const duration = 950
    const startedAt = performance.now()

    setIsTransitioning(true)
    setDisplayConfidence(0)

    const animate = (timestamp: number) => {
      const elapsed = timestamp - startedAt
      const progress = Math.min(elapsed / duration, 1)
      const easedProgress = 1 - Math.pow(1 - progress, 3)

      setDisplayConfidence(Math.round(confidencePercent * easedProgress))

      if (progress < 1) {
        animationFrame = window.requestAnimationFrame(animate)
        return
      }

      setIsTransitioning(false)
    }

    animationFrame = window.requestAnimationFrame(animate)

    return () => {
      window.cancelAnimationFrame(animationFrame)
    }
  }, [confidencePercent, verdict])

  return (
    <Card className="flex h-full min-h-[280px] min-w-0 flex-col overflow-hidden border-[#c7e7f5] bg-white text-[#041E42]">
      <CardHeader>
        <CardTitle className="text-2xl font-bold text-[#041E42] xl:text-3xl">
          {t("verdict")}
        </CardTitle>
      </CardHeader>
      <CardContent
        className={
          isTransitioning
            ? "flex flex-1 flex-col justify-between gap-5 opacity-90 transition-opacity duration-300"
            : "flex flex-1 flex-col justify-between gap-5 opacity-100 transition-opacity duration-300"
        }
      >
        <Badge
          className={
            isViolence
                ? "bg-red-600 px-5 py-2.5 text-xl font-bold text-white hover:bg-red-600"
                : "bg-emerald-600 px-5 py-2.5 text-xl font-bold text-white hover:bg-emerald-600"
          }
        >
          {isViolence ? t("violenceDetected") : t("nonViolence")}
        </Badge>

        <div className="flex flex-1 flex-col justify-center rounded-lg border border-[#c7e7f5] bg-[#f6fbff] p-5">
          <p className={`text-lg font-bold text-[#041E42] ${isArabic ? "text-right" : "text-left"}`}>
            {t("modelConfidence")}
          </p>
          <p
            className={
              isViolence
                ? "mt-3 text-6xl font-extrabold text-red-600"
                : "mt-3 text-6xl font-extrabold text-emerald-600"
            }
          >
            {displayConfidence}%
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
