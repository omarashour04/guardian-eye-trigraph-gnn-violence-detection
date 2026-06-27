import { CheckCircle2, Film } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { type TranslationKey, useLanguage } from "@/context/LanguageContext"
import type { GuardianSample } from "@/types/guardian"

type SampleGalleryProps = {
  samples: GuardianSample[]
  selectedClipId: string
  onSelectSample: (sample: GuardianSample) => void
}

const sampleTranslationKeys: Record<
  string,
  {
    title: TranslationKey
    description: TranslationKey
  }
> = {
  clip_violence_high_001: {
    title: "sampleHighRiskTitle",
    description: "sampleHighRiskDescription",
  },
  clip_non_violence_low_002: {
    title: "sampleNormalTitle",
    description: "sampleNormalDescription",
  },
  clip_violence_medium_003: {
    title: "sampleMediumRiskTitle",
    description: "sampleMediumRiskDescription",
  },
}

export default function SampleGallery({
  samples,
  selectedClipId,
  onSelectSample,
}: SampleGalleryProps) {
  const { t, isArabic } = useLanguage()

  return (
    <Card className="min-w-0 border-[#c7e7f5] bg-white text-[#041E42] shadow-xl shadow-[#0056D2]/10">
      <CardHeader>
        <CardTitle className="text-2xl font-bold text-[#041E42] xl:text-3xl">
          {t("sampleGallery")}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid min-w-0 gap-4 md:grid-cols-2 2xl:grid-cols-3">
          {samples.map((sample) => {
            const isSelected = sample.clip_id === selectedClipId
            const isViolence = sample.verdict === "violence"
            const translatedSample = sampleTranslationKeys[sample.clip_id]
            const title =
              isArabic && translatedSample
                ? t(translatedSample.title)
                : sample.title
            const description =
              isArabic && translatedSample
                ? t(translatedSample.description)
                : sample.description

            return (
              <button
                key={sample.clip_id}
                type="button"
                className={
                  isSelected
                    ? `min-w-0 rounded-xl border border-[#00E5FF] bg-[#E6F7FF] p-3 shadow-lg shadow-[#00A3E0]/20 ${isArabic ? "text-right" : "text-left"}`
                    : `min-w-0 rounded-xl border border-[#c7e7f5] bg-[#f6fbff] p-3 transition hover:border-[#00A3E0]/50 hover:bg-[#E6F7FF] ${isArabic ? "text-right" : "text-left"}`
                }
                onClick={() => onSelectSample(sample)}
              >
                <div className="relative mb-4 flex aspect-video items-center justify-center overflow-hidden rounded-lg border border-[#c7e7f5] bg-[#041E42]">
                  <div className="absolute inset-0 bg-[linear-gradient(rgba(34,211,238,0.07)_1px,transparent_1px),linear-gradient(90deg,rgba(34,211,238,0.07)_1px,transparent_1px)] bg-[size:24px_24px]" />
                  <div className="relative flex size-12 items-center justify-center rounded-full border border-[#00E5FF]/30 bg-[#00A3E0]/20">
                    <Film className="size-5 text-[#00E5FF]" />
                  </div>
                  {isSelected && (
                    <CheckCircle2 className="absolute right-3 top-3 size-5 text-[#00E5FF]" />
                  )}
                </div>

                  <div className={`space-y-3 ${isArabic ? "text-right" : "text-left"}`}>
                  <div className="flex min-w-0 flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                    <h3 className="min-w-0 text-xl font-bold text-[#041E42]">
                      {title}
                    </h3>
                    <Badge
                      className={
                        isViolence
                          ? "bg-red-600 text-white hover:bg-red-600"
                          : "bg-emerald-600 text-white hover:bg-emerald-600"
                      }
                    >
                      {isViolence ? t("violence") : t("nonViolence")}
                    </Badge>
                  </div>

                  <p className="text-base leading-7 text-[#4b647f] xl:text-lg xl:leading-8">
                    {description}
                  </p>

                  <div className="flex items-center justify-between rounded-lg border border-[#c7e7f5] bg-white px-3 py-2">
                    <span className="text-sm font-semibold uppercase text-[#4b647f]">
                      {t("confidence")}
                    </span>
                    <span
                      className={
                        isViolence
                          ? "text-lg font-bold text-red-600"
                          : "text-lg font-bold text-emerald-600"
                      }
                    >
                      {Math.round(sample.confidence * 100)}%
                    </span>
                  </div>
                </div>
              </button>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
