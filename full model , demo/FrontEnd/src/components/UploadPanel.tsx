import { type ChangeEvent, useRef, useState } from "react"
import { Loader2, Upload } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { useLanguage } from "@/context/LanguageContext"

type UploadPanelProps = {
  isLoading?: boolean
  onUpload?: (file: File) => void
}

export default function UploadPanel({
  isLoading = false,
  onUpload,
}: UploadPanelProps) {
  const { t, isArabic } = useLanguage()
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    setSelectedFile(event.target.files?.[0] ?? null)
  }

  const handleUpload = () => {
    if (selectedFile) {
      onUpload?.(selectedFile)
    }
  }

  return (
    <Card className="min-w-0">
      <CardHeader>
        <CardTitle className="text-2xl font-bold text-[#041E42] xl:text-3xl">
          {t("uploadVideo")}
        </CardTitle>
      </CardHeader>
      <CardContent className={`space-y-4 ${isArabic ? "text-right" : "text-left"}`}>
        <Input
          ref={inputRef}
          type="file"
          accept="video/*"
          onChange={handleFileChange}
        />

        <div className="rounded-lg border border-[#c7e7f5] bg-[#f6fbff] p-3">
          <p className="text-sm font-semibold uppercase text-[#4b647f]">
            {t("selectedFile")}
          </p>
          <p
            className={
              selectedFile
                ? "mt-1 break-all text-lg font-semibold leading-8 text-[#041E42]"
                : "mt-1 text-lg leading-8 text-[#4b647f]"
            }
          >
            {selectedFile ? selectedFile.name : t("noVideoSelected")}
          </p>
        </div>

        <Button
          type="button"
          className="w-full text-base font-bold sm:w-auto"
          disabled={!selectedFile || isLoading}
          onClick={handleUpload}
        >
          {isLoading ? (
            <Loader2 className="size-4 animate-spin" />
          ) : (
            <Upload className="size-4" />
          )}
          {isLoading ? t("analyzingVideo") : t("analyzeVideo")}
        </Button>

        {!selectedFile && (
          <p className="rounded-md border border-amber-500/25 bg-amber-50 px-3 py-2 text-base font-semibold leading-7 text-amber-700">
            {t("selectVideoFirst")}
          </p>
        )}
      </CardContent>
    </Card>
  )
}
