import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from 'recharts'
import { getMonitorStatus, postMonitorReset } from '../api/client'
import type { MonitorNoData, MonitorStats, MonitorStatusResponse } from '../api/types'

function isNoData(x: MonitorStatusResponse): x is MonitorNoData {
  return 'status' in x && x.status === 'no_data'
}

type Props = {
  animationsEnabled: boolean
}

function MonitorFrame({
  reduceMotion,
  children,
}: {
  reduceMotion: boolean
  children: ReactNode
}) {
  if (reduceMotion) {
    return <section className="panel">{children}</section>
  }
  return (
    <motion.section
      className="panel"
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.12, ease: 'easeOut' }}
    >
      {children}
    </motion.section>
  )
}

export function MonitorPanel({ animationsEnabled }: Props) {
  const prefersReduced = useReducedMotion()
  const reduceMotion = Boolean(prefersReduced) || !animationsEnabled

  const [data, setData] = useState<MonitorStatusResponse | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [auto, setAuto] = useState(false)
  const [intervalSec, setIntervalSec] = useState(10)

  const refresh = useCallback(async () => {
    setErr(null)
    setLoading(true)
    try {
      setData(await getMonitorStatus())
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'Monitor request failed')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    const t = window.setTimeout(() => {
      void refresh()
    }, 0)
    return () => window.clearTimeout(t)
  }, [refresh])

  useEffect(() => {
    if (!auto) return
    const id = window.setInterval(() => {
      void refresh()
    }, intervalSec * 1000)
    return () => window.clearInterval(id)
  }, [auto, intervalSec, refresh])

  async function onReset() {
    if (!window.confirm('Clear the monitoring rolling window?')) return
    setErr(null)
    try {
      await postMonitorReset()
      await refresh()
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'Reset failed')
    }
  }

  const chartData =
    data && !isNoData(data)
      ? [{ name: 'PSI', value: Math.min(data.psi, 0.35) }]
      : [{ name: 'PSI', value: 0 }]

  const psiFill = (s: MonitorStats) => {
    if (s.psi_status === 'alert') return 'var(--danger)'
    if (s.psi_status === 'warn') return 'var(--warn)'
    return 'var(--ok)'
  }

  return (
    <MonitorFrame reduceMotion={reduceMotion}>
      <h2 className="panel__title">Drift and volume monitor</h2>
      <p className="panel__lead">
        Live stats from <code>GET /monitor/status</code> (PSI vs reference scores, fraud rate in the rolling window,
        optional alerts). Use reset after a controlled retrain.
      </p>
      <div className="row gap wrap">
        <button type="button" className="btn btn--secondary" onClick={() => void refresh()} disabled={loading}>
          {loading ? 'Refreshing…' : 'Refresh now'}
        </button>
        <label className="toggle toggle--switch">
          <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} />
          <span>Auto-refresh</span>
        </label>
        <label className="toggle">
          <span className="panel__muted">Every</span>
          <select
            value={intervalSec}
            onChange={(e) => setIntervalSec(Number(e.target.value))}
            disabled={!auto}
            aria-label="Auto-refresh interval"
          >
            <option value={5}>5 s</option>
            <option value={10}>10 s</option>
            <option value={30}>30 s</option>
          </select>
        </label>
        <button type="button" className="btn btn--ghost btn--danger" onClick={() => void onReset()}>
          Reset window
        </button>
      </div>
      {err && <p className="panel__error">{err}</p>}

      {!data && !err && <p className="panel__muted">Loading monitor…</p>}

      {data && isNoData(data) && <p className="panel__muted">{data.message}</p>}

      {data && !isNoData(data) && (
        <div className="monitor-grid">
          <div className="monitor-chart">
            <h3 className="panel__h3">Population stability (PSI)</h3>
            <p className="panel__muted">Reference threshold at 0.2 (dashed).</p>
            <div className="recharts-wrap">
              <ResponsiveContainer width="100%" height={140}>
                <BarChart layout="vertical" data={chartData} margin={{ top: 8, right: 24, left: 8, bottom: 8 }}>
                  <XAxis type="number" domain={[0, 0.35]} tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                  <YAxis type="category" dataKey="name" width={44} tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                  <ReferenceLine x={0.2} stroke="var(--warn)" strokeDasharray="5 5" />
                  <Bar dataKey="value" radius={[0, 6, 6, 0]} maxBarSize={28}>
                    <Cell fill={psiFill(data)} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="mono psi-readout">
              PSI {data.psi.toFixed(4)} · <span className={`psi-tag psi-tag--${data.psi_status}`}>{data.psi_status}</span>
            </p>
          </div>
          <div>
            <h3 className="panel__h3">Window metrics</h3>
            <dl className="metric-grid metric-grid--tight">
              <div>
                <dt>Predictions (total)</dt>
                <dd>{data.n_total.toLocaleString()}</dd>
              </div>
              <div>
                <dt>Window size</dt>
                <dd>{data.n_window.toLocaleString()}</dd>
              </div>
              <div>
                <dt>Fraud rate (window)</dt>
                <dd>{(100 * data.fraud_rate_window).toFixed(3)}%</dd>
              </div>
              <div>
                <dt>Fraud rate (all time)</dt>
                <dd>{(100 * data.fraud_rate_total).toFixed(3)}%</dd>
              </div>
              <div>
                <dt>Score mean</dt>
                <dd>{data.score_mean_window}</dd>
              </div>
              <div>
                <dt>Score std</dt>
                <dd>{data.score_std_window}</dd>
              </div>
              <div>
                <dt>Score p95</dt>
                <dd>{data.score_p95_window}</dd>
              </div>
            </dl>
          </div>
          <div className="monitor-alerts">
            <h3 className="panel__h3">Alerts</h3>
            {data.alerts.length === 0 ? (
              <p className="panel__muted">No active alerts.</p>
            ) : (
              <ul className="alert-list">
                {data.alerts.map((a, i) => (
                  <motion.li
                    key={`${a.type}-${i}`}
                    className={`alert-list__item alert-list__item--${a.severity}`}
                    initial={reduceMotion ? undefined : { opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.05 }}
                  >
                    <strong>{a.type}</strong> · {a.message}
                  </motion.li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}
    </MonitorFrame>
  )
}
