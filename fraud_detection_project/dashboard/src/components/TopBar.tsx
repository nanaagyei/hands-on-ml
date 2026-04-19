import { motion } from 'framer-motion'
import type { HealthResponse, ModelInfoResponse } from '../api/types'

type Props = {
  health: HealthResponse | null
  modelInfo: ModelInfoResponse | null
  onRefresh: () => void
  busy: boolean
  theme: 'light' | 'dark'
  onThemeChange: (t: 'light' | 'dark') => void
  animationsEnabled: boolean
  onAnimationsChange: (on: boolean) => void
  highContrast: boolean
  onHighContrastChange: (on: boolean) => void
  reduceMotion: boolean
}

export function TopBar({
  health,
  modelInfo,
  onRefresh,
  busy,
  theme,
  onThemeChange,
  animationsEnabled,
  onAnimationsChange,
  highContrast,
  onHighContrastChange,
  reduceMotion,
}: Props) {
  const healthy = health?.status === 'healthy'

  const inner = (
    <>
      <div className="topbar__brand">
        <span className="topbar__title">Fraud operations</span>
        <span className="topbar__subtitle">Model scoring and drift monitor</span>
      </div>
      <div className="topbar__status">
        <span
          className={`status-dot ${healthy ? 'status-dot--ok' : health ? 'status-dot--bad' : 'status-dot--pending'}`}
          aria-hidden
        />
        <span className="topbar__meta">
          API:{' '}
          {health === null
            ? '…'
            : healthy
              ? 'Connected'
              : `Unavailable (${'reason' in health ? health.reason : 'unknown'})`}
        </span>
        {modelInfo && (
          <span className="topbar__meta topbar__meta--dim">
            Model: {modelInfo.model_name}
          </span>
        )}
      </div>
      <div className="topbar__actions">
        <button type="button" className="btn btn--ghost" onClick={onRefresh} disabled={busy}>
          Refresh status
        </button>
        <label className="toggle">
          <span className="sr-only">Theme</span>
          <select
            value={theme}
            onChange={(e) => onThemeChange(e.target.value as 'light' | 'dark')}
            aria-label="Color theme"
          >
            <option value="dark">Dark</option>
            <option value="light">Light</option>
          </select>
        </label>
        <label className="toggle toggle--switch">
          <input
            type="checkbox"
            checked={animationsEnabled}
            onChange={(e) => onAnimationsChange(e.target.checked)}
          />
          <span>Motion</span>
        </label>
        <label className="toggle toggle--switch">
          <input
            type="checkbox"
            checked={highContrast}
            onChange={(e) => onHighContrastChange(e.target.checked)}
          />
          <span>High contrast</span>
        </label>
      </div>
    </>
  )

  if (reduceMotion) {
    return <header className="topbar">{inner}</header>
  }

  return (
    <motion.header
      className="topbar"
      initial={{ opacity: 0, y: -12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: 'easeOut' }}
    >
      {inner}
    </motion.header>
  )
}
