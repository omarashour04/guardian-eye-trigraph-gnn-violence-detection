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
import type { GuardianGate } from "@/types/guardian"

const defaultGate: GuardianGate = {
  skeleton: 0.34,
  interaction: 0.41,
  object: 0.07,
  vit: 0.18,
}

type GateBarProps = {
  gate?: GuardianGate
}

export default function GateBar({ gate = defaultGate }: GateBarProps) {
  const { t, isArabic } = useLanguage()
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
      { name: t("gateSkeleton"), value: animatedGate.skeleton, color: "#22d3ee" },
      { name: t("gateInteraction"), value: animatedGate.interaction, color: "#a78bfa" },
      { name: t("gateObject"), value: animatedGate.object, color: "#f59e0b" },
      { name: t("gateVit"), value: animatedGate.vit, color: "#34d399" },
    ],
    [animatedGate, t],
  )

  return (
    <Card className="flex h-full min-w-0 flex-col border-[#c7e7f5] bg-white text-[#041E42] shadow-sm shadow-blue-100">
      <CardHeader>
        <CardTitle className="text-2xl font-bold text-[#041E42] xl:text-3xl">
          {t("modelGateContributions")}
        </CardTitle>
      </CardHeader>
      <CardContent
        className={
          isTransitioning
            ? "flex flex-1 flex-col justify-between gap-6 opacity-90 transition-opacity duration-300"
            : "flex flex-1 flex-col justify-between gap-6 opacity-100 transition-opacity duration-300"
        }
      >
        <div className="h-72 w-full min-w-0 sm:h-80" dir="ltr">
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
                formatter={(value) => [
                  `${Math.round(Number(value) * 100)}%`,
                  t("contribution"),
                ]}
              />
              <Bar dataKey="value" radius={[0, 8, 8, 0]} barSize={18}>
                {gateData.map((gate) => (
                  <Cell key={gate.name} fill={gate.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          {gateData.map((gate) => (
            <div
              key={gate.name}
              className="rounded-lg border border-[#c7e7f5] bg-[#f6fbff] p-3"
            >
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <span
                    className="size-2.5 rounded-full"
                    style={{ backgroundColor: gate.color }}
                  />
                  <span className="text-base font-bold text-[#041E42]">
                    {gate.name}
                  </span>
                </div>
                <span className="text-lg font-bold text-[#041E42]">
                  {Math.round(gate.value * 100)}%
                </span>
              </div>
              <div className="mt-3 h-2 overflow-hidden rounded-full bg-[#eaf4fb]">
                <div
                  className="h-full rounded-full transition-[width] duration-700 ease-out"
                  style={{
                    width: `${gate.value * 100}%`,
                    backgroundColor: gate.color,
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
