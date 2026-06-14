import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useLanguage } from "@/context/LanguageContext"
import type { GuardianTelemetry } from "@/types/guardian"

const defaultTelemetry: GuardianTelemetry = {
  peak_window: [14, 22],
  total_frames: 40,
  processing_ms: 1260,
  detected_people: 2,
}

type TimelineProps = {
  telemetry?: GuardianTelemetry
}

export default function Timeline({
  telemetry = defaultTelemetry,
}: TimelineProps) {
  const { t } = useLanguage()
  const [startFrame, endFrame] = telemetry.peak_window
  const totalFrames = telemetry.total_frames
  const startPercent = (startFrame / totalFrames) * 100
  const widthPercent = ((endFrame - startFrame) / totalFrames) * 100
  const hasPeakWindow = endFrame > startFrame

  return (
    <Card className="flex h-full min-h-[280px] min-w-0 flex-col border-[#c7e7f5] bg-white text-[#041E42] shadow-sm shadow-blue-100">
      <CardHeader>
        <CardTitle className="text-2xl font-bold text-[#041E42] xl:text-3xl">
          {t("interactionPeakWindow")}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col justify-between gap-6">
        <div className="flex min-w-0 flex-col gap-4">
          <div className="min-w-0">
            <p className="text-2xl font-extrabold text-red-600 xl:text-3xl">
              {hasPeakWindow ? t("frameRange") : t("noPeakWindow")}
            </p>
            {hasPeakWindow && (
              <p
                className="mt-2 text-5xl font-extrabold leading-none text-red-600"
                dir="ltr"
              >
                {startFrame}-{endFrame}
              </p>
            )}
            <p className="mt-2 text-lg font-semibold text-[#4b647f]">
              {hasPeakWindow ? t("highRiskSegment") : t("lowRiskSegment")}
            </p>
          </div>
          <div className="w-fit rounded-lg border border-[#c7e7f5] bg-[#f6fbff] px-4 py-3">
            <p className="text-sm font-bold uppercase text-[#4b647f]">
              {t("frameRange")}
            </p>
            <p className="mt-1 text-lg font-bold text-[#041E42]">
              {t("frameRangeValue")} {totalFrames}
            </p>
          </div>
        </div>

        <div className="space-y-3">
          <div className="relative h-8 overflow-hidden rounded-full border border-[#c7e7f5] bg-[#eaf4fb]">
            <div className="absolute inset-y-0 left-0 w-full bg-[#dff6ff]" />
            {hasPeakWindow && (
              <div
                className="absolute inset-y-1 rounded-full bg-red-500 shadow-lg shadow-red-500/25"
                style={{
                  left: `${startPercent}%`,
                  width: `${widthPercent}%`,
                }}
              />
            )}
          </div>

          <div className="relative h-7 text-sm font-semibold text-[#4b647f]">
            <span className="absolute left-0">0</span>
            {hasPeakWindow && (
              <>
                <span
                  className="absolute -translate-x-1/2 text-red-600"
                  style={{ left: `${startPercent}%` }}
                >
                  {startFrame}
                </span>
                <span
                  className="absolute -translate-x-1/2 text-red-600"
                  style={{ left: `${startPercent + widthPercent}%` }}
                >
                  {endFrame}
                </span>
              </>
            )}
            <span className="absolute right-0">{totalFrames}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
