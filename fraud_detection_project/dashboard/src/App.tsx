import { useCallback, useEffect, useState } from 'react'
import { useReducedMotion } from 'framer-motion'
import { motion } from 'framer-motion'
import { getHealth, getModelInfo } from './api/client'
import type { HealthResponse, ModelInfoResponse } from './api/types'
import { TopBar } from './components/TopBar'
import { ModelPanel } from './components/ModelPanel'
import { SinglePredictPanel } from './components/SinglePredictPanel'
import { BatchPanel } from './components/BatchPanel'
import { MonitorPanel } from './components/MonitorPanel'
import { useDashboardPrefs } from './hooks/useDashboardPrefs'
import './App.css'

export default function App() {
  const {
    theme,
    setTheme,
    animationsEnabled,
    setAnimationsEnabled,
    highContrast,
    setHighContrast,
  } = useDashboardPrefs()

  const prefersReduced = useReducedMotion()
  const reduceMotion = Boolean(prefersReduced) || !animationsEnabled

  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null)
  const [modelErr, setModelErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const refresh = useCallback(async () => {
    setBusy(true)
    setModelErr(null)
    try {
      setHealth(await getHealth())
    } catch (e) {
      setHealth({
        status: 'unhealthy',
        reason: e instanceof Error ? e.message : 'Health check failed',
      })
    }
    try {
      setModelInfo(await getModelInfo())
    } catch (e) {
      setModelInfo(null)
      setModelErr(e instanceof Error ? e.message : 'Failed to load model info')
    } finally {
      setBusy(false)
    }
  }, [])

  useEffect(() => {
    const t = window.setTimeout(() => {
      void refresh()
    }, 0)
    return () => window.clearTimeout(t)
  }, [refresh])

  const Main = reduceMotion ? 'main' : motion.main

  const mainMotion = reduceMotion
    ? {}
    : {
        initial: { opacity: 0 },
        animate: { opacity: 1 },
        transition: { duration: 0.35, delay: 0.08 },
      }

  return (
    <div className="app-shell">
      <TopBar
        health={health}
        modelInfo={modelInfo}
        onRefresh={() => void refresh()}
        busy={busy}
        theme={theme}
        onThemeChange={setTheme}
        animationsEnabled={animationsEnabled}
        onAnimationsChange={setAnimationsEnabled}
        highContrast={highContrast}
        onHighContrastChange={setHighContrast}
        reduceMotion={reduceMotion}
      />
      <Main className="app-main" {...mainMotion}>
        <div className="app-grid">
          <ModelPanel info={modelInfo} error={modelErr} reduceMotion={reduceMotion} />
          <SinglePredictPanel animationsEnabled={animationsEnabled} />
          <BatchPanel />
          <MonitorPanel animationsEnabled={animationsEnabled} />
        </div>
      </Main>
      <footer className="app-footer">
        <span>Fraud detection operator UI · same contract as Flask serving API</span>
      </footer>
    </div>
  )
}
