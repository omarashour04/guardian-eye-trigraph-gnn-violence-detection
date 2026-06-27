import { MapPin } from "lucide-react"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useLanguage } from "@/context/LanguageContext"

export type GuardianCountry = "UK" | "USA California" | "Canada" | "UAE" | "KSA" | "Egypt"

const countryLabels: Record<GuardianCountry, { en: string; ar: string }> = {
  UK: { en: "UK", ar: "المملكة المتحدة" },
  "USA California": { en: "USA California", ar: "الولايات المتحدة - كاليفورنيا" },
  Canada: { en: "Canada", ar: "كندا" },
  UAE: { en: "UAE", ar: "الإمارات" },
  KSA: { en: "KSA", ar: "السعودية" },
  Egypt: { en: "Egypt", ar: "مصر" },
}

const countries = Object.keys(countryLabels) as GuardianCountry[]

type CountrySelectorProps = {
  value: GuardianCountry | ""
  onChange: (country: GuardianCountry | "") => void
}

export default function CountrySelector({ value, onChange }: CountrySelectorProps) {
  const { t, isArabic } = useLanguage()

  return (
    <Card className="min-w-0">
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="flex size-10 items-center justify-center rounded-lg border border-[#00A3E0]/25 bg-[#E6F7FF]">
            <MapPin className="size-5 text-[#0056D2]" />
          </div>
          <CardTitle className="text-2xl font-bold text-[#041E42] xl:text-3xl">
            {t("countrySelector")}
          </CardTitle>
        </div>
      </CardHeader>
      <CardContent className={`space-y-3 ${isArabic ? "text-right" : "text-left"}`}>
        <select
          value={value}
          className="h-11 w-full rounded-lg border border-[#c7e7f5] bg-[#f6fbff] px-3 text-base font-semibold text-[#041E42] outline-none transition focus:border-[#0056D2] focus:ring-2 focus:ring-[#0056D2]/20"
          onChange={(event) => onChange(event.target.value as GuardianCountry | "")}
        >
          <option value="">{t("selectCountry")}</option>
          {countries.map((country) => (
            <option key={country} value={country}>
              {countryLabels[country][isArabic ? "ar" : "en"]}
            </option>
          ))}
        </select>
        <p className="rounded-md border border-[#00A3E0]/25 bg-[#E6F7FF] px-3 py-2 text-sm font-semibold leading-6 text-[#0056D2]">
          {value ? t("legalCountryReady") : t("legalCountryRequired")}
        </p>
      </CardContent>
    </Card>
  )
}
