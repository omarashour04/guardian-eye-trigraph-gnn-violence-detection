import { Globe } from "lucide-react"

import { type Language, useLanguage } from "@/context/LanguageContext"

export default function LanguageSwitcher() {
  const { language, setLanguage, t } = useLanguage()

  return (
    <div className="flex w-fit items-center gap-2 rounded-xl border border-[#00A3E0]/30 bg-[#041E42] px-3 py-2 text-sm shadow-sm shadow-[#00A3E0]/20">
      <Globe className="size-4 text-[#00E5FF]" />
      <label className="sr-only" htmlFor="guardian-eye-language">
        {t("language")}
      </label>
      <select
        id="guardian-eye-language"
        value={language}
        onChange={(event) => setLanguage(event.target.value as Language)}
        className="bg-transparent font-semibold text-white outline-none"
        aria-label={t("language")}
      >
        <option className="text-[#041E42]" value="en">
          {t("english")}
        </option>
        <option className="text-[#041E42]" value="ar">
          {t("arabic")}
        </option>
      </select>
    </div>
  )
}
