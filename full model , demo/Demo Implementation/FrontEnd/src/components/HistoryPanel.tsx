import { Clock, Film } from "lucide-react"
import { useMemo, useState } from "react"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import EmptyState from "@/components/EmptyState"
import LoadingSkeleton from "@/components/LoadingSkeleton"
import { useLanguage } from "@/context/LanguageContext"
import type { GuardianIncident } from "@/types/guardian"

const defaultHistory: GuardianIncident[] = [
  {
    incident_id: "incident_001",
    clip_id: "clip_violence_high_001",
    timestamp: "2026-06-06 15:30",
    verdict: "violence",
    confidence: 0.94,
    thumbnail_url: "/static/thumbnails/clip_001.jpg",
    narrative:
      "High-risk violent interaction detected with strong interaction and skeleton evidence.",
  },
] as const

type HistoryPanelProps = {
  incidents?: GuardianIncident[]
  isLoading?: boolean
  selectedIncidentId?: string
  onSelectIncident?: (incident: GuardianIncident) => void
}

type HistoryThumbnailProps = {
  src?: string
}

function HistoryThumbnail({ src }: HistoryThumbnailProps) {
  const [hasError, setHasError] = useState(false)

  if (!src || hasError) {
    return (
      <div className="flex size-9 items-center justify-center rounded-full border border-[#00E5FF]/30 bg-[#00A3E0]/20">
        <Film className="size-4 text-[#00E5FF]" />
      </div>
    )
  }

  return (
    <img
      src={src}
      alt=""
      className="h-full w-full object-cover"
      loading="lazy"
      onError={() => setHasError(true)}
    />
  )
}

export default function HistoryPanel({
  incidents = defaultHistory,
  isLoading = false,
  selectedIncidentId,
  onSelectIncident,
}: HistoryPanelProps) {
  const { t, isArabic } = useLanguage()
  const sortedIncidents = useMemo(
    () =>
      [...incidents].sort(
        (left, right) =>
          new Date(right.timestamp).getTime() -
          new Date(left.timestamp).getTime(),
      ),
    [incidents],
  )

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp)

    if (Number.isNaN(date.getTime())) {
      return timestamp
    }

    return new Intl.DateTimeFormat(isArabic ? "ar-EG" : "en", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(date)
  }

  return (
    <Card className="flex h-full min-w-0 flex-col border-[#c7e7f5] bg-white text-[#041E42] shadow-xl shadow-[#0056D2]/10">
      <CardHeader className="pb-2">
        <CardTitle className="text-xl font-bold text-[#041E42] xl:text-2xl">
          {t("incidentHistory")}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex min-h-0 flex-1 flex-col">
        {isLoading && <LoadingSkeleton className="min-h-40" />}

        {!isLoading && sortedIncidents.length === 0 && (
          <EmptyState
            icon={<Clock className="size-5" />}
            title={t("noIncidentsRecorded")}
            description={t("populateHistory")}
          />
        )}

        {!isLoading && sortedIncidents.length > 0 && (
          <div className="max-h-[22rem] min-h-0 space-y-2 overflow-y-auto pr-1">
            {sortedIncidents.map((incident) => {
              const isViolence = incident.verdict === "violence"
              const shortIncidentId = incident.incident_id.slice(0, 8)
              const isSelected = selectedIncidentId === incident.incident_id

              return (
                <button
                  key={incident.incident_id}
                  type="button"
                  className={`grid w-full min-w-0 grid-cols-[4.5rem_minmax(0,1fr)] gap-3 rounded-lg border bg-[#f6fbff] p-3 text-left transition ${
                    isSelected
                      ? "border-[#0056D2] ring-2 ring-[#0056D2]/20"
                      : "border-[#c7e7f5] hover:border-[#00A3E0]"
                  } ${isArabic ? "text-right" : "text-left"}`}
                  aria-current={isSelected ? "true" : undefined}
                  disabled={!onSelectIncident}
                  onClick={() => onSelectIncident?.(incident)}
                >
                  <div
                    className="flex aspect-video w-full shrink-0 items-center justify-center overflow-hidden rounded-md border border-[#c7e7f5] bg-[#041E42]"
                    aria-label={`${t("thumbnailPlaceholder")} ${incident.thumbnail_url}`}
                  >
                    <HistoryThumbnail src={incident.thumbnail_url} />
                  </div>

                  <div className="min-w-0">
                    <div className="flex min-w-0 items-start justify-between gap-2">
                      <div className="min-w-0">
                        <p className="truncate text-sm font-extrabold leading-5 text-[#041E42]">
                          {incident.clip_id}
                        </p>
                        <p className="mt-0.5 font-mono text-xs font-semibold uppercase text-[#4b647f]">
                          #{shortIncidentId}
                        </p>
                      </div>
                      <Badge
                        className={
                          isViolence
                            ? "h-6 shrink-0 bg-red-600 px-2 text-xs font-bold text-white hover:bg-red-600"
                            : "h-6 shrink-0 bg-emerald-600 px-2 text-xs font-bold text-white hover:bg-emerald-600"
                        }
                      >
                        {isViolence ? t("violence") : t("nonViolence")}
                      </Badge>
                    </div>

                    <div className="mt-2 flex min-w-0 items-center justify-between gap-2 text-xs font-semibold text-[#4b647f]">
                      <span className="flex min-w-0 items-center gap-1">
                        <Clock className="size-3.5 shrink-0" />
                        <span className="truncate">
                          {formatTimestamp(incident.timestamp)}
                        </span>
                      </span>
                      <span
                        className={
                          isViolence
                            ? "shrink-0 text-sm font-extrabold text-red-600"
                            : "shrink-0 text-sm font-extrabold text-emerald-600"
                        }
                      >
                        {Math.round(incident.confidence * 100)}%
                      </span>
                    </div>
                  </div>
                </button>
              )
            })}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
