import { useEffect, useState } from "react"

import { SplashScreen } from "@/components/SplashScreen"
import DemoPage from "@/pages/DemoPage"

function App() {
  const [showSplash, setShowSplash] = useState(true)

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setShowSplash(false)
    }, 2800)

    return () => window.clearTimeout(timer)
  }, [])

  if (showSplash) {
    return <SplashScreen />
  }

  return <DemoPage />
}

export default App
