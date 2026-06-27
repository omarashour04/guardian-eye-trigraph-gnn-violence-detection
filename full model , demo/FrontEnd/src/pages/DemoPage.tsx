import { useEffect, useState } from "react"
import { Radar, Server, ShieldCheck, Wifi, WifiOff } from "lucide-react"

import logo from "@/assets/gaurdianeye-official-logo-3.png"
import {
  askGuardianEye,
  checkBackendStatus,
  explainIncident,
  getIncidentReview,
  getSamples,
  guardianApiMode,
  predictSample,
  predictVideo,
  type PredictionResponse,
} from "@/api/guardianApi"
import GateBar from "@/components/GateBar"
import OverlayVideo from "@/components/OverlayVideo"
import SampleGallery from "@/components/SampleGallery"
import Timeline from "@/components/Timeline"
import UploadPanel from "@/components/UploadPanel"
import VerdictPanel from "@/components/VerdictPanel"
import CountrySelector, {
  type GuardianCountry,
} from "@/components/CountrySelector"
import DemoNotification from "../components/DemoNotification"
import AskBox from "../components/AskBox"
import HistoryPanel from "../components/HistoryPanel"
import NarrativePanel from "../components/NarrativePanel"
import LegalConsequencesPanel from "@/components/LegalConsequencesPanel"
import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import type {
  GuardianBackendStatus,
  GuardianIncident,
  GuardianSample,
} from "@/types/guardian"
import LanguageSwitcher from "@/components/LanguageSwitcher"
import { useLanguage } from "@/context/LanguageContext"

type NotificationState = {
  type: "success" | "error" | "info" | "warning"
  title: string
  message?: string
}

const stripMockLegalRagForBackendMode = (
  prediction: PredictionResponse,
): PredictionResponse => {
  if (guardianApiMode === "mock") {
    return prediction
  }

  return {
    ...prediction,
    legal_consequences_rag: null,
    legal_scores: null,
  }
}

export default function DemoPage() {
  const { t, isArabic, language } = useLanguage()
  const [samples, setSamples] = useState<GuardianSample[]>([])
  const [selectedSample, setSelectedSample] = useState<GuardianSample | null>(
    null,
  )
  const [selectedIncidentId, setSelectedIncidentId] = useState<string | null>(
    null,
  )
  const [selectedCountry, setSelectedCountry] = useState<GuardianCountry | "">(
    "",
  )
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [isSamplesLoading, setIsSamplesLoading] = useState(true)
  const [isLoading, setIsLoading] = useState(false)
  const [notification, setNotification] = useState<NotificationState | null>(
    null,
  )
  const [backendStatus, setBackendStatus] =
    useState<GuardianBackendStatus>("mock")

  const backendStatusConfig = {
    mock: {
      label: t("mockModeActive"),
      icon: Server,
      className: "border-[#00A3E0]/25 bg-[#E6F7FF] text-[#0056D2]",
    },
    connected: {
      label: t("backendConnected"),
      icon: Wifi,
      className: "border-emerald-500/25 bg-emerald-50 text-emerald-700",
    },
    unavailable: {
      label: t("backendUnavailable"),
      icon: WifiOff,
      className: "border-amber-500/25 bg-amber-50 text-amber-700",
    },
  } satisfies Record<
    GuardianBackendStatus,
    {
      label: string
      icon: typeof Server
      className: string
    }
  >

  const BackendStatusIcon = backendStatusConfig[backendStatus].icon

  useEffect(() => {
    let isMounted = true

    const loadSamples = async () => {
      const loadedSamples = await getSamples()

      if (!isMounted) {
        return
      }

      setSamples(loadedSamples)
      setSelectedSample(loadedSamples[0])
      setPrediction(stripMockLegalRagForBackendMode(loadedSamples[0]))
      setIsSamplesLoading(false)
      setNotification({
        type: "success",
        title: t("sampleLoaded"),
        message: `${loadedSamples[0].title} ${t("sampleReady")}`,
      })
    }

    void loadSamples()

    return () => {
      isMounted = false
    }
  }, [])

  useEffect(() => {
    let isMounted = true

    const loadBackendStatus = async () => {
      const status = await checkBackendStatus()

      if (isMounted) {
        setBackendStatus(status)
      }
    }

    void loadBackendStatus()

    return () => {
      isMounted = false
    }
  }, [])

  useEffect(() => {
    if (!notification) {
      return
    }

    const timeoutId = window.setTimeout(() => {
      setNotification(null)
    }, 3000)

    return () => {
      window.clearTimeout(timeoutId)
    }
  }, [notification])

  const handleSelectSample = async (sample: GuardianSample) => {
    setSelectedSample(sample)
    setSelectedIncidentId(null)
    setIsLoading(true)

    try {
      const result = await predictSample(sample.clip_id)
      setPrediction(stripMockLegalRagForBackendMode(result))
      setNotification({
        type: "success",
        title: t("sampleLoaded"),
        message: `${sample.title} ${t("sampleReady")}`,
      })
    } catch {
      setNotification({
        type: "error",
        title: t("sampleLoadFailed"),
        message: t("sampleLoadFailedMessage"),
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleUpload = async (file: File) => {
    setSelectedIncidentId(null)
    setSelectedSample(null)
    setPrediction(null)
    setIsLoading(true)

    try {
      const result = await predictVideo(file)
      setPrediction(result)

      try {
        const {
          incident_id,
          explanation_rag,
          incident_memory_rag,
          legal_consequences_rag,
          legal_scores,
          ...explanation
        } = await explainIncident(
          result.clip_id,
          language,
          selectedCountry || undefined,
        )
        setSelectedIncidentId(incident_id ?? null)
        setPrediction({
          ...result,
          explanation,
          explanation_rag,
          incident_memory_rag,
          legal_consequences_rag,
          legal_scores,
        })
      } catch {
        setPrediction(result)
      }

      const matchingSample = samples.find(
        (sample) => sample.clip_id === result.clip_id,
      )
      setSelectedSample(matchingSample ?? null)
      setNotification({
        type: "success",
        title: t("videoAnalysisComplete"),
        message: t("videoAnalysisSuccessMessage"),
      })
    } catch {
      setNotification({
        type: "error",
        title: t("videoAnalysisFailed"),
        message: t("videoAnalysisFailedMessage"),
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleReviewIncident = async (incident: GuardianIncident) => {
    setSelectedIncidentId(incident.incident_id)
    setSelectedSample(null)
    setIsLoading(true)

    try {
      const result = await getIncidentReview(
        incident.incident_id,
        language,
        selectedCountry || undefined,
      )
      setPrediction(result)
      setNotification({
        type: "info",
        title: t("incidentHistory"),
        message: incident.clip_id,
      })
    } catch {
      setNotification({
        type: "error",
        title: t("sampleLoadFailed"),
        message: t("sampleLoadFailedMessage"),
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="min-h-screen overflow-x-hidden bg-gradient-to-br from-[#f6fbff] via-[#eef8ff] to-[#dff6ff] text-[#041E42]">
      {notification && (
        <DemoNotification
          type={notification.type}
          title={notification.title}
          message={notification.message}
          onClose={() => setNotification(null)}
        />
      )}

      <div className="mx-auto max-w-7xl space-y-6 px-4 py-6 sm:px-6">
        <section className="min-w-0 rounded-2xl border border-[#c7e7f5] bg-white/90 p-6 shadow-sm shadow-blue-100">
          <div className="flex min-w-0 flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex min-w-0 items-start gap-3">
              <div className="flex size-11 shrink-0 items-center justify-center rounded-lg border border-[#00E5FF]/30 bg-[#E6F7FF]">
                <ShieldCheck className="size-5 text-[#0056D2]" />
              </div>
              <div className={`min-w-0 ${isArabic ? "text-right" : "text-left"}`}>
                <p className="text-lg font-bold uppercase text-[#041E42]">
                  {t("demoModeBanner")}
                </p>
                <p className="mt-1 text-lg font-semibold text-[#4b647f]">
                  {t("classifierStatus")}
                </p>
              </div>
            </div>
            <div className="flex w-fit items-center gap-2 rounded-full border border-[#00A3E0]/25 bg-[#E6F7FF] px-3 py-1.5 text-xs font-medium text-[#0056D2]">
              <Radar className="size-3.5" />
              {t("surveillancePreview")}
            </div>
            <div
              className={`flex w-fit items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium ${backendStatusConfig[backendStatus].className}`}
            >
              <BackendStatusIcon className="size-3.5" />
              {backendStatusConfig[backendStatus].label}
            </div>
          </div>
        </section>

        <header className="rounded-2xl border border-[#c7e7f5] bg-white/80 p-6 shadow-sm shadow-blue-100">
          <div className="flex min-w-0 flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="min-w-0">
            <div className="flex items-center gap-4">
              <img
                src={logo}
                alt="Guardian Eye"
                className="h-24 w-auto md:h-28"
              />

              <div className={isArabic ? "text-right" : "text-left"}>
                <h1 className="text-5xl font-extrabold tracking-wide text-[#041E42] xl:text-6xl">
                  Guardian Eye
                </h1>

                <p className="text-lg font-semibold text-[#4b647f]">
                  AI-Powered Violence Detection & Video Intelligence
                </p>
              </div>
            </div>
          </div>
          <div className="flex w-full flex-col gap-3 sm:w-auto sm:flex-row sm:items-center">
            <LanguageSwitcher />
            <Badge className="w-fit border border-[#00A3E0]/30 bg-[#E6F7FF] px-3 py-1 text-[#0056D2] hover:bg-[#E6F7FF]">
              {t("demoMode")}
            </Badge>
            <div className="w-full rounded-lg border border-[#c7e7f5] bg-[#f6fbff] px-4 py-4 sm:w-auto sm:px-5">
              <p className="text-xs font-medium uppercase text-[#4b647f]">
                {t("surveillanceAi")}
              </p>
              <p className="mt-1 text-sm font-semibold text-[#0056D2]">
                {isLoading
                  ? t("analyzingSelectedVideo")
                  : isSamplesLoading
                    ? t("loadingMockSamples")
                    : t("monitoringPipelineStandby")}
              </p>
            </div>
          </div>
          </div>
        </header>

        <section>
          <Card className="min-w-0 rounded-2xl border-[#c7e7f5] bg-white text-[#041E42] shadow-sm shadow-blue-100">
            <CardHeader>
              <CardTitle className="text-2xl font-bold text-[#041E42] xl:text-3xl">
                {t("dashboardStatus")}
              </CardTitle>
              <CardDescription className="text-lg font-semibold text-[#4b647f]">
                {t("initialFrontendSetup")}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="rounded-md border border-[#00A3E0]/25 bg-[#E6F7FF] px-4 py-3 text-lg font-semibold leading-8 text-[#0056D2]">
                {isLoading
                  ? t("runningAnalysis")
                  : isSamplesLoading
                    ? t("loadingMockDashboardData")
                  : t("dashboardReady")}
              </p>
            </CardContent>
          </Card>
        </section>

        <section className="grid min-w-0 grid-cols-1 gap-6 xl:grid-cols-12">
          <div className="min-w-0 space-y-6 xl:col-span-8 [&>div]:rounded-2xl [&>div]:border-[#c7e7f5] [&>div]:bg-white [&>div]:text-[#041E42] [&>div]:shadow-sm [&>div]:shadow-blue-100">
            <SampleGallery
              samples={samples}
              selectedClipId={selectedSample?.clip_id ?? ""}
              onSelectSample={handleSelectSample}
            />
            <CountrySelector
              value={selectedCountry}
              onChange={setSelectedCountry}
            />
            <UploadPanel isLoading={isLoading} onUpload={handleUpload} />
          </div>
          <div className="grid min-w-0 grid-rows-2 gap-6 xl:col-span-4 [&>div]:h-full [&>div]:rounded-2xl [&>div]:border-[#c7e7f5] [&>div]:bg-white [&>div]:text-[#041E42] [&>div]:shadow-sm [&>div]:shadow-blue-100">
            {prediction && (
              <VerdictPanel
                verdict={prediction.verdict}
                confidence={prediction.confidence}
              />
            )}
            <Timeline telemetry={prediction?.telemetry} />
          </div>
        </section>

        <section className="grid min-w-0 grid-cols-1 items-stretch gap-6 xl:grid-cols-12">
          <div className="min-w-0 xl:col-span-4 [&>div]:h-full [&>div]:rounded-2xl [&>div]:shadow-sm [&>div]:shadow-blue-100">
            <GateBar
              gate={prediction?.gate}
              gateValidity={prediction?.gate_validity}
              inactiveModalities={prediction?.inactive_modalities}
            />
          </div>
          <div className="min-w-0 xl:col-span-8 [&>div]:h-full [&>div]:rounded-2xl [&>div]:shadow-sm [&>div]:shadow-blue-100">
            <OverlayVideo media={prediction?.media} telemetry={prediction?.telemetry} />
          </div>
        </section>

        <section className="grid min-w-0 grid-cols-1 items-stretch gap-6 xl:grid-cols-12">
          <div className="min-w-0 xl:col-span-8 [&>div]:h-full [&>div]:rounded-2xl [&>div]:shadow-sm [&>div]:shadow-blue-100">
            <NarrativePanel
              narrative={prediction?.explanation.narrative}
              narrationMode={prediction?.explanation.narration_mode}
              reasonIfFallback={prediction?.explanation.reason_if_fallback}
            />
          </div>
          <div className="min-w-0 xl:col-span-4 [&>div]:h-full [&>div]:rounded-2xl [&>div]:shadow-sm [&>div]:shadow-blue-100">
            <HistoryPanel
              incidents={prediction?.incidents}
              selectedIncidentId={selectedIncidentId ?? undefined}
              onSelectIncident={handleReviewIncident}
            />
          </div>
        </section>

        <section className="min-w-0 [&>div]:rounded-2xl [&>div]:shadow-sm [&>div]:shadow-blue-100">
          <LegalConsequencesPanel
            legal={prediction?.legal_consequences_rag}
            scores={prediction?.legal_scores}
          />
        </section>

        <section className="min-w-0 [&>div]:rounded-2xl [&>div]:shadow-sm [&>div]:shadow-blue-100">
          <AskBox
            mockAnswer={selectedSample?.ask_response.answer}
            onAsk={(question) =>
              askGuardianEye(
                question,
                {
                  clipId: prediction?.clip_id ?? selectedSample?.clip_id,
                  incidentId: selectedIncidentId,
                  country: selectedCountry || undefined,
                },
                language,
              )
            }
          />
        </section>
      </div>
    </main>
  )
}
