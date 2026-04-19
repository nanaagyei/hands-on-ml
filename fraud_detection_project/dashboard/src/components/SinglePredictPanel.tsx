import { useState } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import { postPredict } from '../api/client'
import type { PredictResult, TransactionInput } from '../api/types'

const SAMPLE_BASE = `${import.meta.env.BASE_URL}samples/`

type Props = {
  animationsEnabled: boolean
}

function parseTransactionJson(raw: string): { ok: true; data: TransactionInput } | { ok: false; error: string } {
  let obj: unknown
  try {
    obj = JSON.parse(raw)
  } catch {
    return { ok: false, error: 'Invalid JSON' }
  }
  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) {
    return { ok: false, error: 'Root must be a JSON object' }
  }
  const o = obj as Record<string, unknown>
  for (let i = 1; i <= 28; i++) {
    const k = `V${i}`
    if (!(k in o)) return { ok: false, error: `Missing ${k}` }
    const n = Number(o[k])
    if (!Number.isFinite(n)) return { ok: false, error: `${k} must be a finite number` }
  }
  if (!('Amount' in o)) return { ok: false, error: 'Missing Amount' }
  if (!('Time' in o)) return { ok: false, error: 'Missing Time' }
  const amount = Number(o.Amount)
  const time = Number(o.Time)
  if (!Number.isFinite(amount) || amount < 0) return { ok: false, error: 'Amount must be a number ≥ 0' }
  if (!Number.isFinite(time) || time < 0) return { ok: false, error: 'Time must be a number ≥ 0' }
  const data = { ...o } as unknown as TransactionInput
  return { ok: true, data }
}

export function SinglePredictPanel({ animationsEnabled }: Props) {
  const prefersReduced = useReducedMotion()
  const reduceMotion = Boolean(prefersReduced) || !animationsEnabled

  const [json, setJson] = useState('')
  const [parseErr, setParseErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [apiErr, setApiErr] = useState<string | null>(null)
  const [result, setResult] = useState<PredictResult | null>(null)

  async function loadSample(name: 'legit' | 'fraud') {
    setParseErr(null)
    setApiErr(null)
    setResult(null)
    try {
      const res = await fetch(`${SAMPLE_BASE}${name}.json`)
      if (!res.ok) throw new Error(`Sample not found (${res.status})`)
      const text = await res.text()
      setJson(text)
    } catch (e) {
      setParseErr(e instanceof Error ? e.message : 'Failed to load sample')
    }
  }

  async function onSubmit() {
    setApiErr(null)
    setResult(null)
    const parsed = parseTransactionJson(json)
    if (!parsed.ok) {
      setParseErr(parsed.error)
      return
    }
    setParseErr(null)
    setLoading(true)
    try {
      const r = await postPredict(parsed.data)
      setResult(r)
    } catch (e) {
      setApiErr(e instanceof Error ? e.message : 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  const Bar = reduceMotion ? 'div' : motion.div

  return (
    <section className="panel">
      <h2 className="panel__title">Score one transaction</h2>
      <p className="panel__lead">
        Paste JSON with <code>V1</code>–<code>V28</code>, <code>Amount</code>, and <code>Time</code>. Values are
        validated locally before calling <code>POST /predict</code>.
      </p>
      <div className="row gap">
        <button type="button" className="btn btn--secondary" onClick={() => loadSample('legit')}>
          Load sample (typical)
        </button>
        <button type="button" className="btn btn--secondary" onClick={() => loadSample('fraud')}>
          Load sample (stressed)
        </button>
      </div>
      <textarea
        className="input input--textarea"
        rows={10}
        value={json}
        onChange={(e) => setJson(e.target.value)}
        placeholder='{ "V1": 0.0, ... "V28": 0.0, "Amount": 10.5, "Time": 0 }'
        spellCheck={false}
        aria-label="Transaction JSON"
      />
      {parseErr && <p className="panel__error">{parseErr}</p>}
      {apiErr && <p className="panel__error">{apiErr}</p>}
      <button type="button" className="btn btn--primary" onClick={onSubmit} disabled={loading || !json.trim()}>
        {loading ? 'Scoring…' : 'Score transaction'}
      </button>

      {result && (
        <div className="result-block">
          <div className={`verdict ${result.is_fraud ? 'verdict--fraud' : 'verdict--ok'}`}>
            {result.is_fraud ? 'Flagged as fraud' : 'Not flagged'}
          </div>
          <div className="risk-meter">
            <div className="risk-meter__labels">
              <span>Normalized risk</span>
              <span>{(result.risk_score_norm * 100).toFixed(1)}%</span>
            </div>
            <div className="risk-meter__track">
              <Bar
                className="risk-meter__fill"
                initial={reduceMotion ? undefined : { width: 0 }}
                animate={{ width: `${Math.min(100, result.risk_score_norm * 100)}%` }}
                transition={{ duration: reduceMotion ? 0 : 0.55, ease: 'easeOut' }}
              />
            </div>
          </div>
          <dl className="result-dl">
            <div>
              <dt>Risk score</dt>
              <dd>{result.risk_score.toFixed(4)}</dd>
            </div>
            <div>
              <dt>Threshold</dt>
              <dd>{result.threshold.toFixed(4)}</dd>
            </div>
            <div>
              <dt>Confidence</dt>
              <dd className={`badge badge--${result.confidence}`}>{result.confidence}</dd>
            </div>
            <div>
              <dt>Amount</dt>
              <dd>{result.amount.toFixed(2)}</dd>
            </div>
            <div>
              <dt>Latency</dt>
              <dd>{result.latency_ms.toFixed(2)} ms</dd>
            </div>
          </dl>
          <p className="explanation">{result.explanation}</p>
        </div>
      )}
    </section>
  )
}
