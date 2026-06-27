import { FileText } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { useLanguage } from "@/context/LanguageContext"

const narrative =
  "The model detected a violent interaction primarily from interaction and skeleton signals. The highest activity occurred between frames 14 and 22. The confidence score indicates a strong likelihood of violent behavior."

type NarrativePanelProps = {
  narrative?: string
  narrationMode?: "vlm_llm" | "vlm_only" | "fallback"
  reasonIfFallback?: string | null
}

export default function NarrativePanel({
  narrative: incidentNarrative = narrative,
  narrationMode,
  reasonIfFallback,
}: NarrativePanelProps) {
  const { t, isArabic } = useLanguage()
  const modeLabel =
    narrationMode === "vlm_llm" || narrationMode === "vlm_only"
      ? "AI-generated"
      : "Fallback summary"

  return (
    <Card className="flex h-full min-w-0 flex-col border-[#c7e7f5] bg-white text-[#041E42] shadow-xl shadow-[#0056D2]/10">
      <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex size-10 items-center justify-center rounded-lg border border-[#00A3E0]/25 bg-[#E6F7FF]">
            <FileText className="size-5 text-[#0056D2]" />
          </div>
          <CardTitle className="text-2xl font-bold text-[#041E42] xl:text-3xl">
            {t("incidentNarrative")}
          </CardTitle>
        </div>
        <Badge
          className={
            narrationMode === "vlm_llm"
            || narrationMode === "vlm_only"
              ? "w-fit border border-emerald-500/30 bg-emerald-50 px-3 py-1 text-emerald-700 hover:bg-emerald-50"
              : "w-fit border border-[#00A3E0]/30 bg-[#E6F7FF] px-3 py-1 text-[#0056D2] hover:bg-[#E6F7FF]"
          }
          title={reasonIfFallback ?? undefined}
        >
          {narrationMode ? modeLabel : t("aiExplanation")}
        </Badge>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col">
        <div
          className="rounded-lg border border-[#c7e7f5] bg-[#f6fbff] p-4 sm:p-5"
          dir={isArabic ? "rtl" : "ltr"}
        >
          <p
            className={`max-w-none whitespace-pre-line text-xl font-medium leading-9 text-[#041E42] ${
              isArabic ? "text-right" : "text-left"
            }`}
          >
            {incidentNarrative}
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
