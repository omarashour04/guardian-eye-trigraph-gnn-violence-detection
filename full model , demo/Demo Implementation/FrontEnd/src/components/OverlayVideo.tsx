import { Radio } from "lucide-react"

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { useLanguage } from "@/context/LanguageContext"
import type { GuardianMedia } from "@/types/guardian"

type OverlayMedia = {
  original_video_url?: string
  overlay_video_url?: string
  thumbnail_url?: string
} & Partial<Pick<GuardianMedia, "source" | "resolution" | "mode">>

const defaultMedia: OverlayMedia = {
  source: "Mock overlay stream",
  resolution: "720p",
  mode: "Demo",
}

type OverlayVideoProps = {
  media?: OverlayMedia
}

export default function OverlayVideo({
  media = defaultMedia,
}: OverlayVideoProps) {
  const { t } = useLanguage()
  const playbackUrl = media.overlay_video_url ?? media.original_video_url
  const overlayStatus = media.overlay_video_url
    ? t("overlayReady")
    : playbackUrl || media.thumbnail_url
      ? t("previewPending")
      : t("overlayStandby")

  const metadata = [
    { label: t("source"), value: media.source ?? "Mock overlay stream" },
    { label: t("resolution"), value: media.resolution ?? "720p" },
    { label: t("mode"), value: media.mode ?? "Demo" },
    { label: t("overlayStatus"), value: overlayStatus },
  ]

  return (
    <Card className="dark-panel flex h-full min-w-0 flex-col overflow-hidden shadow-sm shadow-blue-100">
      <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <CardTitle className="text-2xl font-bold text-white xl:text-3xl">
            {t("overlayVideo")}
          </CardTitle>
          <CardDescription className="mt-1 text-lg font-semibold text-white/75">
            {t("backendOverlayPlaceholder")}
          </CardDescription>
        </div>
        <div className="flex w-fit items-center gap-2 rounded-full border border-[#00E5FF]/30 bg-[#00A3E0]/20 px-3 py-1 text-xs font-medium text-[#00E5FF]">
          <Radio className="size-3.5" />
          {overlayStatus}
        </div>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col gap-5">
        <div className="relative aspect-video min-h-52 overflow-hidden rounded-lg border border-[#00E5FF]/25 bg-[#041E42] sm:min-h-0">
          {playbackUrl ? (
            <video
              className="h-full w-full bg-black object-cover"
              controls
              preload="metadata"
            >
              <source src={playbackUrl} type="video/mp4" />
            </video>
          ) : media.thumbnail_url ? (
            <>
              <div
                className="absolute inset-0 bg-cover bg-center opacity-30"
                style={{ backgroundImage: `url(${media.thumbnail_url})` }}
              />
              <div className="absolute inset-0 bg-[#041E42]/75" />
              <div className="absolute inset-0 bg-[linear-gradient(rgba(34,211,238,0.08)_1px,transparent_1px),linear-gradient(90deg,rgba(34,211,238,0.08)_1px,transparent_1px)] bg-[size:32px_32px]" />
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(14,165,233,0.16),transparent_44%)]" />
              <div className="relative flex h-full flex-col items-center justify-center px-4 text-center sm:px-6">
                <div className="mb-4 flex size-12 items-center justify-center rounded-full border border-[#00E5FF]/30 bg-[#00A3E0]/20">
                  <Radio className="size-5 text-[#00E5FF]" />
                </div>
                <p className="max-w-md text-lg font-bold leading-8 text-white">
                  {t("overlayPreviewPending")}
                </p>
                <p className="mt-2 max-w-md text-sm font-semibold uppercase text-white/60">
                  {t("thumbnailOverlayAwaiting")}
                </p>
              </div>
            </>
          ) : (
            <>
              <div className="absolute inset-0 bg-[linear-gradient(rgba(34,211,238,0.08)_1px,transparent_1px),linear-gradient(90deg,rgba(34,211,238,0.08)_1px,transparent_1px)] bg-[size:32px_32px]" />
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(14,165,233,0.14),transparent_42%)]" />
              <div className="absolute inset-x-0 top-1/2 h-px bg-cyan-300/30 shadow-[0_0_18px_rgba(34,211,238,0.45)]" />
              <div className="relative flex h-full flex-col items-center justify-center px-4 text-center sm:px-6">
                <div className="mb-4 flex size-12 items-center justify-center rounded-full border border-[#00E5FF]/30 bg-[#00A3E0]/20">
                  <Radio className="size-5 text-[#00E5FF]" />
                </div>
                <p className="max-w-md text-lg font-bold leading-8 text-white">
                  {t("overlayVideoPending")}
                </p>
                <p className="mt-2 text-sm font-semibold uppercase text-white/60">
                  {t("awaitingProcessedStream")}
                </p>
              </div>
            </>
          )}
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
              <p className="mt-1 text-base font-bold text-white">
                {item.value}
              </p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
