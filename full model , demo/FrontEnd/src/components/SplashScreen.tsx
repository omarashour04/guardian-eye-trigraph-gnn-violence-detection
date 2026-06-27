import logo from "@/assets/gaurdianeye-official-logo-3.png"
import { useLanguage } from "@/context/LanguageContext"

export function SplashScreen() {
  const { t } = useLanguage()

  return (
    <div className="flex min-h-screen items-center justify-center overflow-hidden bg-[#041E42]">
      <div className="relative flex flex-col items-center px-6 text-center">
        <div className="absolute inset-0 rounded-full bg-[#00E5FF]/20 blur-3xl animate-pulse" />

        <img
          src={logo}
          alt="Guardian Eye"
          className="relative z-10 w-80 animate-logo-glow md:w-[30rem] lg:w-[36rem]"
        />

        <p className="relative z-10 mt-6 text-sm uppercase tracking-[0.35em] text-white/70 md:text-base">
          {t("initializing")}
        </p>
      </div>
    </div>
  )
}
