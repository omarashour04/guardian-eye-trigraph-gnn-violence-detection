import {
  createContext,
  type ReactNode,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react"

export type Language = "en" | "ar"

const translations = {
  en: {
    aiExplanation: "AI Explanation",
    analyzeVideo: "Analyze Video",
    analyzingSelectedVideo: "Analyzing selected video",
    analyzingVideo: "Analyzing video...",
    arabic: "العربية",
    askGuardian: "Ask Guardian Eye",
    askGuardianPlaceholder: "Ask about detected incidents...",
    backendConnected: "Backend Connected",
    backendOverlayPlaceholder: "Backend overlay placeholder",
    backendUnavailable: "Backend Unavailable",
    classifierStatus: "Classifier active • Mock backend responses • Defense Preview",
    closeNotification: "Close notification",
    confidence: "Confidence",
    dashboardReady: "Frontend dashboard structure is ready",
    dashboardStatus: "Dashboard Status",
    demoMode: "Demo Mode",
    demoModeBanner: "DEMO MODE - Guardian Eye Violence Detection System",
    english: "English",
    frameRange: "Frame Range",
    frameRangeValue: "0 to",
    gateInteraction: "Interaction",
    gateObject: "Object",
    gateSkeleton: "Skeleton",
    gateVit: "ViT",
    gateUnavailable: "N/A",
    gateValidityPartial:
      "Some modality inputs were unavailable. Inactive stream percentages are hidden to avoid misleading contribution claims.",
    gateValidityUnknown:
      "Gate validity metadata is unavailable for this prediction.",
    guardianResponse: "Guardian Eye Response",
    highRiskSegment: "High-risk segment",
    incidentHistory: "Incident History",
    incidentNarrative: "Incident Narrative",
    initializing: "Initializing Surveillance Intelligence",
    interactionPeakWindow: "Interaction Peak Window",
    language: "Language",
    loadingMockSamples: "Loading local mock samples",
    loadingMockDashboardData: "Loading mock dashboard data...",
    lowRiskSegment: "Low-risk segment",
    mode: "Mode",
    mockModeActive: "Mock Mode Active",
    modelConfidence: "Model Confidence",
    modelGateContributions: "Model Gate Contributions",
    monitoringPipelineStandby: "Monitoring pipeline standby",
    noPeakWindow: "No peak window",
    noIncidentsRecorded: "No incidents recorded yet",
    nonViolence: "Non-Violence",
    overlayPreviewPending: "Overlay preview pending",
    overlayReady: "Overlay ready",
    overlayStandby: "Overlay standby",
    overlayStatus: "Overlay status",
    overlayVideo: "Overlay Video",
    overlayVideoPending:
      "Overlay video will appear here after backend processing",
    previewPending: "Preview pending",
    queryPastIncidents: "Query past incidents and model explanations",
    resolution: "Resolution",
    reviewingContext: "Reviewing incident context...",
    runningAnalysis: "Running violence detection analysis...",
    sampleGallery: "Sample Gallery",
    sampleHighRiskDescription:
      "Two-person confrontation with strong motion and contact cues.",
    sampleHighRiskTitle: "High-Risk Corridor Incident",
    sampleLoadFailed: "Sample load failed",
    sampleLoaded: "Sample loaded",
    sampleMediumRiskDescription:
      "Brief aggressive movement with moderate interaction evidence.",
    sampleMediumRiskTitle: "Medium-Risk Entrance Event",
    sampleNormalDescription:
      "Low-risk passage through monitored area with no conflict cues.",
    sampleNormalTitle: "Normal Lobby Movement",
    sampleReady: "is ready for review.",
    selectVideoFirst: "Select a video file before running analysis.",
    selectedFile: "Selected file",
    source: "Source",
    thumbnailPlaceholder: "Thumbnail placeholder",
    surveillanceAi: "Surveillance AI",
    surveillancePreview: "Surveillance preview",
    thinking: "Thinking...",
    thumbnailOverlayAwaiting:
      "Thumbnail received, processed overlay awaiting backend",
    uploadVideo: "Upload Video",
    video: "Video",
    videoAnalysisComplete: "Analysis complete",
    videoAnalysisFailed: "Analysis failed",
    videoAnalysisFailedMessage:
      "Guardian Eye could not complete the mock video analysis.",
    videoAnalysisSuccessMessage: "Mock video analysis finished successfully.",
    violence: "violence",
    violenceDetected: "Violence Detected",
    sampleLoadFailedMessage: "Guardian Eye could not load this mock sample.",
    noVideoSelected: "No video selected",
    awaitingProcessedStream: "Awaiting processed detection stream",
    contribution: "Contribution",
    country: "Country",
    countrySelector: "Legal Country",
    emptyLegalReferences:
      "Not enough retrieved legal references for a grounded legal consequence summary.",
    guardrailStatus: "Guardrail Status",
    initialFrontendSetup: "Initial frontend setup",
    legalCountryReady: "Legal context will use the selected country.",
    legalCountryRequired:
      "Legal Consequences requires a country. Classifier and explanation can still run.",
    legalGuardrailWarning:
      "The legal summary needs review and should not be presented confidently.",
    legalReferences: "Legal References",
    legalSelectCountryPrompt: "Select a country to request possible legal consequences.",
    limitationsNote: "Limitations Note",
    needsReview: "Needs review",
    notAvailable: "Not available",
    notSelected: "Not selected",
    overallScore: "Overall Score",
    possibleLegalConsequences: "Possible Legal Consequences",
    ragMode: "RAG Mode",
    populateHistory:
      "Analyze a video or load a sample to populate the incident history.",
    reference: "Reference",
    selectCountry: "Select country",
    askGuardianButton: "Ask Guardian Eye",
    verdict: "Guardian Eye Verdict",
  },
  ar: {
    aiExplanation: "شرح الذكاء الاصطناعي",
    analyzeVideo: "تحليل الفيديو",
    analyzingSelectedVideo: "جار تحليل الفيديو المحدد",
    analyzingVideo: "جار تحليل الفيديو...",
    arabic: "العربية",
    askGuardian: "اسأل Guardian Eye",
    askGuardianPlaceholder: "اسأل عن الحوادث المكتشفة...",
    backendConnected: "الخادم متصل",
    backendOverlayPlaceholder: "عنصر توضيحي مؤقت من الخادم",
    backendUnavailable: "الخادم غير متاح",
    classifierStatus: "المصنف نشط • ردود خلفية تجريبية • معاينة العرض",
    closeNotification: "إغلاق الإشعار",
    confidence: "الثقة",
    dashboardReady: "هيكل لوحة التحكم الأمامية جاهز",
    dashboardStatus: "حالة النظام",
    demoMode: "وضع العرض",
    demoModeBanner: "وضع العرض - نظام Guardian Eye لاكتشاف العنف",
    english: "English",
    frameRange: "نطاق الإطارات",
    frameRangeValue: "0 إلى",
    gateInteraction: "التفاعل",
    gateObject: "العنصر",
    gateSkeleton: "الهيكل",
    gateVit: "ViT",
    gateUnavailable: "غير متاح",
    gateValidityPartial:
      "بعض مدخلات الأنماط غير متاحة. تم إخفاء نسب التيارات غير النشطة لتجنب عرض مساهمات مضللة.",
    gateValidityUnknown:
      "بيانات صلاحية مساهمات البوابات غير متاحة لهذا التنبؤ.",
    guardianResponse: "رد Guardian Eye",
    highRiskSegment: "مقطع عالي الخطورة",
    incidentHistory: "سجل الحوادث",
    incidentNarrative: "سرد الحادثة",
    initializing: "تهيئة ذكاء المراقبة",
    interactionPeakWindow: "نافذة ذروة التفاعل",
    language: "اللغة",
    loadingMockSamples: "جار تحميل العينات التجريبية المحلية",
    loadingMockDashboardData: "جار تحميل بيانات لوحة التحكم التجريبية...",
    lowRiskSegment: "مقطع منخفض الخطورة",
    mode: "الوضع",
    mockModeActive: "وضع المحاكاة نشط",
    modelConfidence: "ثقة النموذج",
    modelGateContributions: "مساهمات بوابات النموذج",
    monitoringPipelineStandby: "خط المراقبة في وضع الاستعداد",
    noPeakWindow: "لا توجد نافذة ذروة",
    noIncidentsRecorded: "لا توجد حوادث مسجلة بعد",
    nonViolence: "غير عنيف",
    overlayPreviewPending: "معاينة التوضيح قيد الانتظار",
    overlayReady: "التوضيح جاهز",
    overlayStandby: "التوضيح في وضع الاستعداد",
    overlayStatus: "حالة التوضيح",
    overlayVideo: "الفيديو التوضيحي",
    overlayVideoPending: "سيظهر الفيديو التوضيحي هنا بعد معالجة الخادم",
    previewPending: "المعاينة قيد الانتظار",
    queryPastIncidents: "استعلم عن الحوادث السابقة وتفسيرات النموذج",
    resolution: "الدقة",
    reviewingContext: "جار مراجعة سياق الحادثة...",
    runningAnalysis: "جار تشغيل تحليل اكتشاف العنف...",
    sampleGallery: "معرض العينات",
    sampleHighRiskDescription:
      "مواجهة بين شخصين مع مؤشرات قوية للحركة والاحتكاك.",
    sampleHighRiskTitle: "حادثة ممر عالية الخطورة",
    sampleLoadFailed: "فشل تحميل العينة",
    sampleLoaded: "تم تحميل العينة",
    sampleMediumRiskDescription:
      "حركة عدوانية قصيرة مع دليل تفاعل متوسط.",
    sampleMediumRiskTitle: "حدث مدخل متوسط الخطورة",
    sampleNormalDescription:
      "مرور منخفض الخطورة داخل منطقة مراقبة دون مؤشرات صراع.",
    sampleNormalTitle: "حركة طبيعية في الردهة",
    sampleReady: "جاهزة للمراجعة.",
    selectVideoFirst: "يرجى اختيار ملف فيديو قبل بدء التحليل.",
    selectedFile: "الملف المحدد",
    source: "المصدر",
    thumbnailPlaceholder: "عنصر صورة مصغرة",
    surveillanceAi: "ذكاء المراقبة",
    surveillancePreview: "معاينة المراقبة",
    thinking: "جار التفكير...",
    thumbnailOverlayAwaiting:
      "تم استلام الصورة المصغرة، والتوضيح المعالج بانتظار الخادم",
    uploadVideo: "رفع فيديو",
    video: "الفيديو",
    videoAnalysisComplete: "اكتمل التحليل",
    videoAnalysisFailed: "فشل التحليل",
    videoAnalysisFailedMessage:
      "تعذر على Guardian Eye إكمال تحليل الفيديو التجريبي.",
    videoAnalysisSuccessMessage: "اكتمل تحليل الفيديو التجريبي بنجاح.",
    violence: "عنيف",
    violenceDetected: "تم اكتشاف عنف",
    sampleLoadFailedMessage: "تعذر على Guardian Eye تحميل هذه العينة التجريبية.",
    noVideoSelected: "لم يتم اختيار فيديو",
    awaitingProcessedStream: "بانتظار بث الاكتشاف المعالج",
    contribution: "المساهمة",
    initialFrontendSetup: "الإعداد الأولي للواجهة الأمامية",
    populateHistory: "حلل فيديو أو حمل عينة لملء سجل الحوادث.",
    country: "الدولة",
    countrySelector: "الدولة القانونية",
    emptyLegalReferences:
      "لا توجد مراجع قانونية مسترجعة كافية لملخص قانوني مؤسس.",
    guardrailStatus: "حالة الحماية",
    legalCountryReady: "سيستخدم السياق القانوني الدولة المحددة.",
    legalCountryRequired:
      "العواقب القانونية تتطلب اختيار دولة. يمكن أن يستمر التصنيف والشرح.",
    legalGuardrailWarning:
      "الملخص القانوني يحتاج إلى مراجعة ولا يجب عرضه بثقة عالية.",
    legalReferences: "المراجع القانونية",
    legalSelectCountryPrompt: "اختر دولة لطلب العواقب القانونية المحتملة.",
    limitationsNote: "ملاحظة القيود",
    needsReview: "يحتاج مراجعة",
    notAvailable: "غير متاح",
    notSelected: "غير محدد",
    overallScore: "الدرجة الكلية",
    possibleLegalConsequences: "العواقب القانونية المحتملة",
    ragMode: "وضع RAG",
    reference: "مرجع",
    selectCountry: "اختر الدولة",
    askGuardianButton: "اسأل Guardian Eye",
    verdict: "نتيجة تحليل Guardian Eye",
  },
} as const

export type TranslationKey = keyof typeof translations.en

type LanguageContextValue = {
  language: Language
  setLanguage: (language: Language) => void
  t: (key: TranslationKey) => string
  isArabic: boolean
}

const LanguageContext = createContext<LanguageContextValue | null>(null)

const getInitialLanguage = (): Language => {
  if (typeof window === "undefined") {
    return "en"
  }

  const storedLanguage = window.localStorage.getItem("guardian-eye-language")
  return storedLanguage === "ar" || storedLanguage === "en"
    ? storedLanguage
    : "en"
}

type LanguageProviderProps = {
  children: ReactNode
}

export function LanguageProvider({ children }: LanguageProviderProps) {
  const [language, setLanguage] = useState<Language>(getInitialLanguage)
  const isArabic = language === "ar"

  useEffect(() => {
    window.localStorage.setItem("guardian-eye-language", language)
    document.documentElement.lang = language
    document.documentElement.dir = isArabic ? "rtl" : "ltr"
  }, [isArabic, language])

  const value = useMemo(
    () => ({
      language,
      setLanguage,
      t: (key: TranslationKey) => translations[language][key],
      isArabic,
    }),
    [isArabic, language],
  )

  return (
    <LanguageContext.Provider value={value}>
      {children}
    </LanguageContext.Provider>
  )
}

export function useLanguage() {
  const context = useContext(LanguageContext)

  if (!context) {
    throw new Error("useLanguage must be used within LanguageProvider")
  }

  return context
}
