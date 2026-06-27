import { useEffect, useMemo, useState } from "react"

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useLanguage } from "@/context/LanguageContext"
import type { GuardianGate, GuardianGateValidity } from "@/types/guardian"

const defaultGate: GuardianGate = {
  skeleton: 0.34,
  interaction: 0.41,
  object: 0.07,
  vit: 0.18,
}

type GateBarProps = {
  gate?: GuardianGate
  gateValidity?: GuardianGateValidity
  inactiveModalities?: string[]
}

export default function GateBar({
  gate = defaultGate,
  gateValidity,
  inactiveModalities = [],
}: GateBarProps) {
  const { t, isArabic } = useLanguage()
  const inactiveSet = useMemo(
    () => new Set([...(inactiveModalities ?? []), ...(gateValidity?.unavailable_contributions ?? [])]),
    [gateValidity?.unavailable_contributions, inactiveModalities],
  )
  const warningInactiveSet = useMemo(
    () => new Set([...inactiveSet].filter((name) => name !== "vit")),
    [inactiveSet],
  )
  const [animatedGate, setAnimatedGate] = useState<GuardianGate>({
    skeleton: 0,
    interaction: 0,
    object: 0,
    vit: 0,
  })
  const [isTransitioning, setIsTransitioning] = useState(false)

  useEffect(() => {
    let animationFrame = 0
    const duration = 1000
    const startedAt = performance.now()

    setIsTransitioning(true)
    setAnimatedGate({
      skeleton: 0,
      interaction: 0,
      object: 0,
      vit: 0,
    })

    const animate = (timestamp: number) => {
      const elapsed = timestamp - startedAt
      const progress = Math.min(elapsed / duration, 1)
      const easedProgress = 1 - Math.pow(1 - progress, 3)

      setAnimatedGate({
        skeleton: gate.skeleton * easedProgress,
        interaction: gate.interaction * easedProgress,
        object: gate.object * easedProgress,
        vit: gate.vit * easedProgress,
      })

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
  }, [gate.skeleton, gate.interaction, gate.object, gate.vit])

  const gateData = useMemo(
    () => [
      {
        key: "skeleton",
        name: t("gateSkeleton"),
        value: animatedGate.skeleton,
        rawValue: gate.skeleton,
        color: "#22d3ee",
        active: !inactiveSet.has("skeleton"),
      },
      {
        key: "interaction",
        name: t("gateInteraction"),
        value: animatedGate.interaction,
        rawValue: gate.interaction,
        color: "#a78bfa",
        active: !inactiveSet.has("interaction"),
      },
      {
        key: "object",
        name: t("gateObject"),
        value: animatedGate.object,
        rawValue: gate.object,
        color: "#f59e0b",
        active: !inactiveSet.has("object"),
      },
      {
        key: "vit",
        name: inactiveSet.has("vit")
          ? isArabic
            ? "ViT / RGB بوابة خام"
            : "ViT / RGB raw gate"
          : t("gateVit"),
        value: animatedGate.vit,
        rawValue: gate.vit,
        color: "#34d399",
        active: true,
        rawGateOnly: inactiveSet.has("vit"),
      },
    ],
    [animatedGate, gate, inactiveSet, isArabic, t],
  )
  const validityMessage =
    gateValidity?.status === "partial" && warningInactiveSet.size > 0
      ? gateValidity.message || t("gateValidityPartial")
      : gateValidity?.status === "unknown"
        ? gateValidity.message || t("gateValidityUnknown")
        : null

  return (
    <Card className="flex h-full min-h-0 min-w-0 flex-col border-[#c7e7f5] bg-white text-[#041E42] shadow-sm shadow-blue-100">
      <CardHeader className="shrink-0 pb-3">
        <CardTitle className="text-2xl font-bold text-[#041E42] xl:text-3xl">
          {t("modelGateContributions")}
        </CardTitle>
      </CardHeader>
      <CardContent
        className={
          isTransitioning
            ? "flex min-h-0 flex-1 flex-col gap-4 opacity-90 transition-opacity duration-300"
            : "flex min-h-0 flex-1 flex-col gap-4 opacity-100 transition-opacity duration-300"
        }
      >
        {validityMessage && (
          <div className="shrink-0 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm font-semibold leading-snug text-amber-900">
            {validityMessage}
          </div>
        )}
        <div className="min-h-[260px] w-full min-w-0 flex-1" dir="ltr">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={gateData}
              layout="vertical"
              margin={{ top: 8, right: 20, bottom: 8, left: isArabic ? 28 : 12 }}
            >
              <CartesianGrid
                horizontal={false}
                stroke="#9bcfe6"
                strokeDasharray="3 3"
              />
              <XAxis
                type="number"
                domain={[0, 1]}
                tickFormatter={(value) => `${Math.round(value * 100)}%`}
                stroke="#041E42"
                tick={{ fill: "#041E42", fontSize: 14, fontWeight: 600 }}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                type="category"
                dataKey="name"
                width={isArabic ? 118 : 106}
                stroke="#041E42"
                tick={{
                  fill: "#041E42",
                  fontSize: 14,
                  fontWeight: 700,
                }}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip
                cursor={{ fill: "rgba(15, 23, 42, 0.72)" }}
                contentStyle={{
                  background: "#020617",
                  border: "1px solid #1e293b",
                  borderRadius: "8px",
                  color: "#f8fafc",
                }}
                formatter={(value, _name, item) => [
                  item.payload.active
                    ? `${Math.round(Number(value) * 100)}%`
                    : t("gateUnavailable"),
                  t("contribution"),
                ]}
              />
              <Bar dataKey="value" radius={[0, 8, 8, 0]} barSize={18}>
                {gateData.map((gate) => (
                  <Cell
                    key={gate.name}
                    fill={gate.active ? gate.color : "#94a3b8"}
                    fillOpacity={gate.active ? 1 : 0.45}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="grid shrink-0 gap-3 sm:grid-cols-2">
          {gateData.map((gate) => (
            <div
              key={gate.name}
              className="min-w-0 rounded-lg border border-[#c7e7f5] bg-[#f6fbff] p-3"
            >
              <div className="flex items-center justify-between gap-3">
                <div className="flex min-w-0 items-center gap-2">
                  <span
                    className="size-2.5 shrink-0 rounded-full"
                    style={{ backgroundColor: gate.active ? gate.color : "#94a3b8" }}
                  />
                  <span className="min-w-0 break-words text-base font-bold leading-tight text-[#041E42]">
                    {gate.name}
                  </span>
                </div>
                <span className="shrink-0 text-lg font-bold text-[#041E42]">
                  {gate.active ? `${Math.round(gate.value * 100)}%` : t("gateUnavailable")}
                </span>
              </div>
              <div className="mt-3 h-2 overflow-hidden rounded-full bg-[#eaf4fb]">
                <div
                  className="h-full rounded-full transition-[width] duration-700 ease-out"
                  style={{
                    width: gate.active ? `${gate.value * 100}%` : "100%",
                    backgroundColor: gate.active ? gate.color : "#cbd5e1",
                  }}
                />
              </div>
              {!gate.active && (
                <p className="mt-2 text-xs font-semibold text-slate-500">
                  raw gate {Math.round(gate.rawValue * 100)}%
                </p>
              )}
              {gate.rawGateOnly && (
                <p className="mt-2 text-xs font-semibold text-emerald-700">
                  {isArabic ? "عرض نسبة البوابة الخام" : "Raw gate percentage shown"}
                </p>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
