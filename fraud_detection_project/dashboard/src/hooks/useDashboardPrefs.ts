import { useCallback, useEffect, useState } from 'react'

type Theme = 'light' | 'dark'

const STORAGE = {
  theme: 'fraud-dash-theme',
  motion: 'fraud-dash-motion',
  contrast: 'fraud-dash-contrast',
} as const

function readStoredString(key: string, fallback: string): string {
  try {
    return localStorage.getItem(key) ?? fallback
  } catch {
    return fallback
  }
}

export function useDashboardPrefs() {
  const [theme, setThemeState] = useState<Theme>(() => {
    const s = readStoredString(STORAGE.theme, '')
    if (s === 'light' || s === 'dark') return s
    if (typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: light)').matches)
      return 'light'
    return 'dark'
  })

  const [animationsEnabled, setAnimationsEnabled] = useState(
    () => readStoredString(STORAGE.motion, '1') !== '0',
  )

  const [highContrast, setHighContrast] = useState(
    () => readStoredString(STORAGE.contrast, '0') === '1',
  )

  const setTheme = useCallback((t: Theme) => {
    setThemeState(t)
    try {
      localStorage.setItem(STORAGE.theme, t)
    } catch {
      /* ignore */
    }
  }, [])

  const setAnimationsEnabledPersist = useCallback((on: boolean) => {
    setAnimationsEnabled(on)
    try {
      localStorage.setItem(STORAGE.motion, on ? '1' : '0')
    } catch {
      /* ignore */
    }
  }, [])

  const setHighContrastPersist = useCallback((on: boolean) => {
    setHighContrast(on)
    try {
      localStorage.setItem(STORAGE.contrast, on ? '1' : '0')
    } catch {
      /* ignore */
    }
  }, [])

  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  useEffect(() => {
    document.documentElement.dataset.contrast = highContrast ? 'high' : 'normal'
  }, [highContrast])

  useEffect(() => {
    document.documentElement.dataset.motion =
      animationsEnabled ? 'full' : 'reduced'
  }, [animationsEnabled])

  return {
    theme,
    setTheme,
    animationsEnabled,
    setAnimationsEnabled: setAnimationsEnabledPersist,
    highContrast,
    setHighContrast: setHighContrastPersist,
  }
}
