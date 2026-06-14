import { ExternalLink, Scale, ShieldAlert } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useLanguage } from "@/context/LanguageContext"
import type {
  GuardianLegalConsequencesRag,
  GuardianLegalScores,
} from "@/types/guardian"

type LegalConsequencesPanelProps = {
  legal?: GuardianLegalConsequencesRag | null
  scores?: GuardianLegalScores | null
}

const unavailableSourceLinkText = {
  en: "No external source link available.",
  ar: "لا يوجد رابط خارجي متاح.",
}

const isExternalHttpUrl = (url?: string | null) =>
  typeof url === "string" && /^https?:\/\//i.test(url)

const safetyNote = {
  en: "This is not legal advice, does not determine guilt, and does not predict court outcome.",
  ar: "هذه ليست استشارة قانونية، ولا تحدد الإدانة، ولا تتنبأ بنتيجة المحكمة.",
}

export default function LegalConsequencesPanel({
  legal,
  scores,
}: LegalConsequencesPanelProps) {
  const { t, isArabic } = useLanguage()
  const references = legal?.retrieved_legal_references ?? []
  const hasReferences = references.length > 0
  const isPassed = legal?.guardrail_status === "passed"
  const summary = legal?.summary ?? t("legalSelectCountryPrompt")
  const source = legal?.legal_rag_source ?? "fallback"

  return (
    <Card className="flex h-full min-w-0 flex-col border-[#c7e7f5] bg-white text-[#041E42] shadow-xl shadow-[#0056D2]/10">
      <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex size-10 items-center justify-center rounded-lg border border-[#00A3E0]/25 bg-[#E6F7FF]">
            <Scale className="size-5 text-[#0056D2]" />
          </div>
          <CardTitle className="text-2xl font-bold text-[#041E42] xl:text-3xl">
            {t("possibleLegalConsequences")}
          </CardTitle>
        </div>
        <Badge
          className={
            source === "real" && isPassed
              ? "w-fit border border-emerald-500/30 bg-emerald-50 px-3 py-1 text-emerald-700 hover:bg-emerald-50"
              : "w-fit border border-amber-500/30 bg-amber-50 px-3 py-1 text-amber-700 hover:bg-amber-50"
          }
        >
          {source}
        </Badge>
      </CardHeader>
      <CardContent className={`space-y-4 ${isArabic ? "text-right" : "text-left"}`}>
        <div
          className={
            isPassed
              ? "rounded-lg border border-[#c7e7f5] bg-[#f6fbff] p-4"
              : "rounded-lg border border-amber-500/30 bg-amber-50 p-4"
          }
        >
          <p className="text-lg font-semibold leading-8 text-[#041E42]">
            {summary}
          </p>
          {!isPassed && (
            <p className="mt-3 flex items-start gap-2 text-sm font-semibold leading-6 text-amber-700">
              <ShieldAlert className="mt-1 size-4 shrink-0" />
              <span>{t("legalGuardrailWarning")}</span>
            </p>
          )}
        </div>

        <div className="grid gap-3 md:grid-cols-3">
          <div className="rounded-lg border border-[#c7e7f5] bg-white p-3">
            <p className="text-xs font-bold uppercase text-[#4b647f]">
              {t("country")}
            </p>
            <p className="mt-1 text-base font-extrabold text-[#041E42]">
              {legal?.country || t("notSelected")}
            </p>
          </div>
          <div className="rounded-lg border border-[#c7e7f5] bg-white p-3">
            <p className="text-xs font-bold uppercase text-[#4b647f]">
              {t("guardrailStatus")}
            </p>
            <p className="mt-1 text-base font-extrabold text-[#041E42]">
              {legal?.guardrail_status ?? t("needsReview")}
            </p>
          </div>
          <div className="rounded-lg border border-[#c7e7f5] bg-white p-3">
            <p className="text-xs font-bold uppercase text-[#4b647f]">
              {t("ragMode")}
            </p>
            <p className="mt-1 text-base font-extrabold text-[#041E42]">
              {legal?.rag_mode ?? t("notAvailable")}
            </p>
          </div>
        </div>

        {legal?.legal_rag_warning && (
          <p className="rounded-lg border border-amber-500/30 bg-amber-50 p-3 text-sm font-semibold leading-6 text-amber-700">
            {legal.legal_rag_warning}
          </p>
        )}

        {scores?.overall_score !== null && scores?.overall_score !== undefined && (
          <div className="rounded-lg border border-[#c7e7f5] bg-white p-3">
            <p className="text-xs font-bold uppercase text-[#4b647f]">
              {t("overallScore")}
            </p>
            <p className="mt-1 text-base font-extrabold text-[#041E42]">
              {Math.round(scores.overall_score * 100)}%
            </p>
          </div>
        )}

        <div className="rounded-lg border border-[#c7e7f5] bg-[#f6fbff] p-4">
          <p className="text-sm font-bold uppercase text-[#4b647f]">
            {t("limitationsNote")}
          </p>
          <p className="mt-2 text-base font-semibold leading-7 text-[#041E42]">
            {legal?.limitations_note || safetyNote[isArabic ? "ar" : "en"]}
          </p>
        </div>

        <div className="space-y-3">
          <div className="flex items-center justify-between gap-3">
            <p className="text-sm font-bold uppercase text-[#4b647f]">
              {t("legalReferences")}
            </p>
            <Badge className="border border-[#00A3E0]/30 bg-[#E6F7FF] text-[#0056D2] hover:bg-[#E6F7FF]">
              {references.length}
            </Badge>
          </div>

          {!hasReferences && (
            <p className="rounded-lg border border-amber-500/30 bg-amber-50 p-3 text-sm font-semibold leading-6 text-amber-700">
              {t("emptyLegalReferences")}
            </p>
          )}

          {references.map((reference, index) => (
            <article
              key={`${reference.law_title}-${reference.article_number ?? index}`}
              className="rounded-lg border border-[#c7e7f5] bg-white p-4"
            >
              <div className="flex min-w-0 flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                <div className="min-w-0">
                  <h3 className="break-words text-base font-extrabold text-[#041E42]">
                    {reference.law_title}
                  </h3>
                  <p className="mt-1 text-sm font-semibold text-[#4b647f]">
                    {[reference.article_number, reference.section_title]
                      .filter(Boolean)
                      .join(" · ") || t("reference")}
                  </p>
                </div>
                <Badge className="w-fit border border-[#00A3E0]/30 bg-[#E6F7FF] text-[#0056D2] hover:bg-[#E6F7FF]">
                  {Math.round(reference.score * 100)}%
                </Badge>
              </div>
              <p className="mt-3 text-sm font-medium leading-6 text-[#041E42]">
                {reference.snippet}
              </p>
              {isExternalHttpUrl(reference.source_url) ? (
                <a
                  href={reference.source_url}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-3 inline-flex items-center gap-1 text-sm font-bold text-[#0056D2] hover:underline"
                >
                  <ExternalLink className="size-4" />
                  {t("source")}
                </a>
              ) : (
                <p className="mt-3 text-sm font-bold text-[#4b647f]">
                  {unavailableSourceLinkText[isArabic ? "ar" : "en"]}
                </p>
              )}
            </article>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
