import { useState, type ReactNode } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import type { ModelInfoResponse } from '../api/types'

type Props = {
  info: ModelInfoResponse | null
  error: string | null
  reduceMotion: boolean
}

function PanelFrame({
  reduceMotion,
  className,
  children,
}: {
  reduceMotion: boolean
  className: string
  children: ReactNode
}) {
  if (reduceMotion) {
    return <section className={className}>{children}</section>
  }
  return (
    <motion.section
      className={className}
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.05, ease: 'easeOut' }}
    >
      {children}
    </motion.section>
  )
}

export function ModelPanel({ info, error, reduceMotion }: Props) {
  const [open, setOpen] = useState(false)

  if (error) {
    return (
      <PanelFrame reduceMotion={reduceMotion} className="panel panel--error">
        <h2 className="panel__title">Model metadata</h2>
        <p className="panel__error">{error}</p>
      </PanelFrame>
    )
  }

  if (!info) {
    return (
      <PanelFrame reduceMotion={reduceMotion} className="panel">
        <h2 className="panel__title">Model metadata</h2>
        <p className="panel__muted">Loading…</p>
      </PanelFrame>
    )
  }

  return (
    <PanelFrame reduceMotion={reduceMotion} className="panel">
      <h2 className="panel__title">Model metadata</h2>
      <p className="panel__lead">
        Threshold {info.threshold.toFixed(4)}
        {info.threshold_label ? ` · ${info.threshold_label}` : ''} · {info.n_features}{' '}
        features
      </p>
      <dl className="metric-grid">
        <div>
          <dt>Test AUC-PR</dt>
          <dd>{info.test_auc_pr != null ? info.test_auc_pr.toFixed(4) : '—'}</dd>
        </div>
        <div>
          <dt>Test recall</dt>
          <dd>{info.test_recall != null ? info.test_recall.toFixed(4) : '—'}</dd>
        </div>
        <div>
          <dt>Test precision</dt>
          <dd>{info.test_precision != null ? info.test_precision.toFixed(4) : '—'}</dd>
        </div>
        <div>
          <dt>Train fraud rate</dt>
          <dd>
            {info.fraud_rate_train != null ? `${(100 * info.fraud_rate_train).toFixed(3)}%` : '—'}
          </dd>
        </div>
      </dl>
      <button type="button" className="btn btn--link" onClick={() => setOpen((o) => !o)}>
        {open ? 'Hide' : 'Show'} feature names ({info.feature_names.length})
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            className="feature-list-wrap"
            initial={reduceMotion ? false : { height: 0, opacity: 0 }}
            animate={reduceMotion ? {} : { height: 'auto', opacity: 1 }}
            exit={reduceMotion ? {} : { height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
          >
            <ul className="feature-list">
              {info.feature_names.map((f) => (
                <li key={f}>
                  <code>{f}</code>
                </li>
              ))}
            </ul>
          </motion.div>
        )}
      </AnimatePresence>
    </PanelFrame>
  )
}
