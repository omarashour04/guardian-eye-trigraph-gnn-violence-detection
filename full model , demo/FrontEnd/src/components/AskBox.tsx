import { type FormEvent, useEffect, useState } from "react"
import { MessageSquareText, Send } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { useLanguage } from "@/context/LanguageContext"
import type { GuardianAskResponse } from "@/types/guardian"

const defaultMockAnswer =
  "I found one violent incident. It involved two people with strong interaction and skeleton signals. The peak activity occurred between frames 14 and 22."

type AskBoxProps = {
  mockAnswer?: string
  onAsk?: (question: string) => Promise<GuardianAskResponse>
}

export default function AskBox({
  mockAnswer: sampleAnswer = defaultMockAnswer,
  onAsk,
}: AskBoxProps) {
  const { t, isArabic } = useLanguage()
  const [question, setQuestion] = useState("")
  const [answer, setAnswer] = useState("")
  const [modeLabel, setModeLabel] = useState<string | null>(null)
  const [fallbackReason, setFallbackReason] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    setAnswer("")
  }, [sampleAnswer])

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()

    if (!question.trim() || isLoading) {
      return
    }

    setIsLoading(true)
    setAnswer("")
    setModeLabel(null)
    setFallbackReason(null)

    try {
      if (onAsk) {
        const response = await onAsk(question)
        setAnswer(response.answer)
        setModeLabel(
          response.grounding_label
            ? response.grounding_label
            : response.ask_mode === "llm"
            ? "LLM"
            : response.ask_mode === "grounded"
              ? isArabic ? "السجل الموثق" : "Grounded history"
              : "Fallback",
        )
        setFallbackReason(response.reason_if_fallback ?? null)
        return
      }

      await new Promise((resolve) => window.setTimeout(resolve, 700))
      setAnswer(sampleAnswer)
      setModeLabel("Fallback")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Card className="min-w-0 border-[#c7e7f5] bg-white text-[#041E42] shadow-xl shadow-[#0056D2]/10">
      <CardHeader>
        <div className="flex min-w-0 items-start gap-3">
          <div className="flex size-10 shrink-0 items-center justify-center rounded-lg border border-[#00A3E0]/25 bg-[#E6F7FF]">
            <MessageSquareText className="size-5 text-[#0056D2]" />
          </div>
          <div className="min-w-0">
            <CardTitle className="text-2xl font-bold text-[#041E42] xl:text-3xl">
              {t("askGuardian")}
            </CardTitle>
            <CardDescription className="mt-1 text-lg font-semibold text-[#4b647f]">
              {t("queryPastIncidents")}
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <form className="space-y-4" onSubmit={handleSubmit}>
          <Textarea
            className={`min-h-28 border-[#c7e7f5] bg-[#f6fbff] text-lg leading-8 text-[#041E42] placeholder:text-[#4b647f] ${isArabic ? "text-right" : "text-left"}`}
            placeholder={t("askGuardianPlaceholder")}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
          />

          <Button
            type="submit"
            className="w-full bg-[#0056D2] text-base font-bold text-white hover:bg-[#00A3E0] sm:w-auto"
            disabled={!question.trim() || isLoading}
          >
            <Send className="size-4" />
            {isLoading ? t("thinking") : t("askGuardianButton")}
          </Button>
        </form>

        {(isLoading || answer) && (
          <div className="mt-5 rounded-lg border border-[#c7e7f5] bg-[#f6fbff] p-4">
            <p className="text-sm font-bold uppercase text-[#4b647f]">
              {isArabic ? "إجابة Guardian Eye" : t("guardianResponse")}
            </p>
            {!isLoading && modeLabel && (
              <p
                className="mt-1 text-xs font-bold uppercase text-[#0056D2]"
                title={fallbackReason ?? undefined}
              >
                {modeLabel}
              </p>
            )}
            <p className="mt-2 whitespace-pre-line break-words text-lg font-medium leading-8 text-[#041E42]">
              {isLoading ? t("reviewingContext") : answer}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
