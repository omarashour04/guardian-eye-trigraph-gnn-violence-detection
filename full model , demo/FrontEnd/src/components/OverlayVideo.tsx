import { useMemo, useState } from "react"
import {
  ChevronDown,
  ChevronUp,
  Maximize2,
  Minimize2,
  Radio,
} from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { useLanguage } from "@/context/LanguageContext"
import type { GuardianMedia, GuardianTelemetry } from "@/types/guardian"

type OverlayStreamKey = "skeleton" | "interaction" | "object" | "vit"

type OverlayMedia = {
  original_video_url?: string
  overlay_video_url?: string
  thumbnail_url?: string
  overlays?: GuardianMedia["overlays"]
  overlay_status?: GuardianMedia["overlay_status"]
  cache_key?: string
} & Partial<Pick<GuardianMedia, "source" | "resolution" | "mode">>

type StreamConfig = {
  key: OverlayStreamKey
  title: string
  detail: string
}

type StreamSlot = StreamConfig & {
  playbackUrl?: string
  thumbnailUrl?: string
  status: "ready" | "pending" | "standby"
  statusLabel?: string
  pendingNote?: string
}

const streamConfigs: StreamConfig[] = [
  {
    key: "skeleton",
    title: "Skeleton Stream",
    detail: "Pose-only overlay",
  },
  {
    key: "interaction",
    title: "Interaction Stream",
    detail: "People boxes and relationship cues",
  },
  {
    key: "object",
    title: "Object Stream",
    detail: "Object and person-object cues",
  },
  {
    key: "vit",
    title: "ViT / RGB Stream",
    detail: "Raw appearance stream",
  },
]

const defaultMedia: OverlayMedia = {
  source: "Mock overlay stream",
  resolution: "720p",
  mode: "Demo",
}

type OverlayVideoProps = {
  media?: OverlayMedia
  telemetry?: GuardianTelemetry
}

const streamStatusLabel = {
  ready: "stream overlay ready",
  pending: "stream overlay pending",
  standby: "awaiting stream",
} satisfies Record<StreamSlot["status"], string>

export default function OverlayVideo({
  media = defaultMedia,
  telemetry,
}: OverlayVideoProps) {
  const { t, isArabic } = useLanguage()
  const [expandedStream, setExpandedStream] =
    useState<OverlayStreamKey | null>(null)
  const [collapsedStreams, setCollapsedStreams] = useState<
    Partial<Record<OverlayStreamKey, boolean>>
  >({})

  const combinedOverlayUrl = media.overlay_video_url
  const sourceUrl = media.original_video_url
  const hasStreamMap = media.overlays !== undefined
  const slots = useMemo(
    () =>
      streamConfigs.map<StreamSlot>((stream) => {
        const streamUrl = media.overlays?.[stream.key] ?? undefined
        const backendStatus = media.overlay_status?.[stream.key]
        const isVit = stream.key === "vit"
        const fallbackUrl = isVit ? sourceUrl : combinedOverlayUrl ?? sourceUrl
        const playbackUrl = streamUrl ?? (hasStreamMap ? undefined : fallbackUrl)
        const status = streamUrl
          ? "ready"
          : backendStatus === "missing"
            ? "standby"
          : hasStreamMap
            ? "pending"
            : playbackUrl
              ? "pending"
              : "standby"

        return {
          ...stream,
          playbackUrl,
          thumbnailUrl: media.thumbnail_url,
          status,
          statusLabel:
            backendStatus === "available"
              ? "available"
              : backendStatus === "fallback_placeholder"
                ? "fallback placeholder"
                : backendStatus === "missing"
                  ? "missing"
                  : undefined,
          pendingNote: hasStreamMap
            ? "backend did not return this stream yet"
            : "using combined overlay until backend provides this stream",
        }
      }),
    [
      combinedOverlayUrl,
      hasStreamMap,
      media.overlays,
      media.thumbnail_url,
      sourceUrl,
    ],
  )

  const overlayStatus = combinedOverlayUrl
    ? t("overlayReady")
    : sourceUrl || media.thumbnail_url
      ? t("previewPending")
      : t("overlayStandby")

  const metadata = [
    { label: t("source"), value: media.source ?? "Mock overlay stream" },
    { label: t("resolution"), value: media.resolution ?? "720p" },
    { label: t("mode"), value: media.mode ?? "Demo" },
    { label: t("overlayStatus"), value: overlayStatus },
  ]

  const toggleCollapsed = (stream: OverlayStreamKey) => {
    setCollapsedStreams((current) => ({
      ...current,
      [stream]: !current[stream],
    }))
  }

  const toggleExpanded = (stream: OverlayStreamKey) => {
    setExpandedStream((current) => (current === stream ? null : stream))
  }

  return (
    <Card className="dark-panel flex h-full min-w-0 flex-col overflow-hidden shadow-sm shadow-blue-100">
      <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <CardTitle className="text-2xl font-bold text-white xl:text-3xl">
            {t("overlayVideo")}
          </CardTitle>
          <CardDescription className="mt-1 text-lg font-semibold text-white/75">
            Four model streams with stream-specific slots
          </CardDescription>
        </div>
        <div className="flex w-fit items-center gap-2 rounded-full border border-[#00E5FF]/30 bg-[#00A3E0]/20 px-3 py-1 text-xs font-medium text-[#00E5FF]">
          <Radio className="size-3.5" />
          {overlayStatus}
        </div>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col gap-5">
        <div className="grid min-w-0 gap-4 lg:grid-cols-2">
          {slots.map((slot) => {
            const isCollapsed = Boolean(collapsedStreams[slot.key])
            const isExpanded = expandedStream === slot.key
            const videoKey = [
              slot.key,
              media.source,
              media.cache_key,
              slot.playbackUrl,
            ].join("|")

            return (
              <div
                key={slot.key}
                className={`min-w-0 rounded-lg border border-[#00E5FF]/20 bg-white/10 p-3 ${
                  isExpanded ? "lg:col-span-2" : ""
                }`}
              >
                <div className="flex min-w-0 items-start justify-between gap-3">
                  <div className="min-w-0">
                    <h3 className="text-lg font-bold text-white">
                      {slot.title}
                    </h3>
                    <p className="mt-1 text-sm font-semibold text-white/65">
                      {slot.detail}
                    </p>
                  </div>
                  <div className="flex shrink-0 items-center gap-1">
                    <Button
                      type="button"
                      size="icon"
                      variant="ghost"
                      className="size-8 text-white/80 hover:bg-white/10 hover:text-white"
                      onClick={() => toggleExpanded(slot.key)}
                      title={isExpanded ? "Restore size" : "Maximize stream"}
                    >
                      {isExpanded ? (
                        <Minimize2 className="size-4" />
                      ) : (
                        <Maximize2 className="size-4" />
                      )}
                    </Button>
                    <Button
                      type="button"
                      size="icon"
                      variant="ghost"
                      className="size-8 text-white/80 hover:bg-white/10 hover:text-white"
                      onClick={() => toggleCollapsed(slot.key)}
                      title={isCollapsed ? "Expand stream" : "Collapse stream"}
                    >
                      {isCollapsed ? (
                        <ChevronDown className="size-4" />
                      ) : (
                        <ChevronUp className="size-4" />
                      )}
                    </Button>
                  </div>
                </div>

                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <span
                    className={`rounded-full border px-2.5 py-1 text-xs font-bold uppercase ${
                      slot.status === "ready"
                        ? "border-emerald-300/30 bg-emerald-400/15 text-emerald-100"
                        : slot.status === "pending"
                          ? "border-amber-300/30 bg-amber-300/15 text-amber-100"
                          : "border-white/20 bg-white/10 text-white/70"
                    }`}
                  >
                    {slot.statusLabel ?? streamStatusLabel[slot.status]}
                  </span>
                  {slot.status === "pending" && (
                    <span className="text-xs font-semibold text-white/55">
                      {slot.pendingNote}
                    </span>
                  )}
                </div>

                {!isCollapsed && (
                  <StreamPreview
                    key={videoKey}
                    slot={slot}
                    videoKey={videoKey}
                    telemetry={telemetry}
                    peakLabel={isArabic ? "نافذة الذروة" : "Peak window"}
                  />
                )}
              </div>
            )
          })}
        </div>

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {metadata.map((item) => (
            <div
              key={item.label}
              className="rounded-lg border border-[#00E5FF]/20 bg-white/10 p-3"
            >
              <p className="text-sm font-semibold uppercase text-white/60">
                {item.label}
              </p>
              <p className="mt-1 break-words text-base font-bold text-white">
                {item.value}
              </p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

type StreamPreviewProps = {
  slot: StreamSlot
  videoKey: string
  telemetry?: GuardianTelemetry
  peakLabel: string
}

function StreamPreview({ slot, videoKey, telemetry, peakLabel }: StreamPreviewProps) {
  const [playback, setPlayback] = useState({ currentTime: 0, duration: 0 })
  const isPeakActive = isInsidePeakWindow(
    playback.currentTime,
    playback.duration,
    telemetry,
  )

  return (
    <div
      className={`relative mt-3 aspect-video min-h-48 overflow-hidden rounded-lg border bg-[#041E42] transition-[border-color,box-shadow] sm:min-h-0 ${
        isPeakActive
          ? "border-4 border-red-500 shadow-[0_0_0_2px_rgba(254,202,202,0.95),0_0_28px_rgba(239,68,68,0.9)]"
          : "border-4 border-[#00E5FF]/25"
      }`}
    >
      {isPeakActive && (
        <div className="pointer-events-none absolute left-3 top-3 z-10 rounded-md border border-red-200 bg-red-600 px-2.5 py-1 text-xs font-black uppercase tracking-wide text-white shadow-lg shadow-red-950/40">
          {peakLabel}
        </div>
      )}
      {slot.playbackUrl ? (
        <video
          key={videoKey}
          className="h-full w-full bg-black object-cover"
          controls
          preload="metadata"
          onLoadedMetadata={(event) =>
            setPlayback({
              currentTime: event.currentTarget.currentTime,
              duration: event.currentTarget.duration || 0,
            })
          }
          onTimeUpdate={(event) =>
            setPlayback({
              currentTime: event.currentTarget.currentTime,
              duration: event.currentTarget.duration || 0,
            })
          }
        >
          <source src={slot.playbackUrl} type="video/mp4" />
        </video>
      ) : slot.thumbnailUrl ? (
        <>
          <div
            className="absolute inset-0 bg-cover bg-center opacity-30"
            style={{ backgroundImage: `url(${slot.thumbnailUrl})` }}
          />
          <div className="absolute inset-0 bg-[#041E42]/75" />
          <StreamPlaceholder title="Stream preview pending" />
        </>
      ) : (
        <StreamPlaceholder title="Awaiting processed stream" />
      )}
    </div>
  )
}

function isInsidePeakWindow(
  currentTime: number,
  duration: number,
  telemetry?: GuardianTelemetry,
) {
  if (!telemetry || !duration || !Number.isFinite(duration)) {
    return false
  }

  const [startFrame, endFrame] = telemetry.peak_window
  if (startFrame === endFrame && startFrame === 0) {
    return false
  }

  const totalFrames = Math.max(Number(telemetry.total_frames || 0), endFrame, 32)
  const startSeconds = (Math.max(startFrame, 0) / totalFrames) * duration
  const endSeconds = (Math.max(endFrame, startFrame) / totalFrames) * duration
  return currentTime >= startSeconds && currentTime <= endSeconds
}

function StreamPlaceholder({ title }: { title: string }) {
  return (
    <>
      <div className="absolute inset-0 bg-[linear-gradient(rgba(34,211,238,0.08)_1px,transparent_1px),linear-gradient(90deg,rgba(34,211,238,0.08)_1px,transparent_1px)] bg-[size:32px_32px]" />
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(14,165,233,0.14),transparent_42%)]" />
      <div className="relative flex h-full flex-col items-center justify-center px-4 text-center sm:px-6">
        <div className="mb-4 flex size-12 items-center justify-center rounded-full border border-[#00E5FF]/30 bg-[#00A3E0]/20">
          <Radio className="size-5 text-[#00E5FF]" />
        </div>
        <p className="max-w-md text-lg font-bold leading-8 text-white">
          {title}
        </p>
      </div>
    </>
  )
}
