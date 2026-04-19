import { Fragment, useMemo, useState } from 'react'
import { postPredictBatch } from '../api/client'
import type { BatchPredictResponse, TransactionInput } from '../api/types'

function parseBatchJson(raw: string): { ok: true; data: TransactionInput[] } | { ok: false; error: string } {
  let obj: unknown
  try {
    obj = JSON.parse(raw)
  } catch {
    return { ok: false, error: 'Invalid JSON' }
  }
  if (!obj || typeof obj !== 'object' || !Array.isArray((obj as { transactions?: unknown }).transactions)) {
    return { ok: false, error: 'Expected { "transactions": [ {...}, ... ] }' }
  }
  const txs = (obj as { transactions: unknown[] }).transactions
  if (txs.length === 0) return { ok: false, error: 'transactions array is empty' }
  if (txs.length > 1000) return { ok: false, error: 'Maximum 1000 rows per batch' }
  for (let i = 0; i < txs.length; i++) {
    const row = txs[i]
    if (!row || typeof row !== 'object') return { ok: false, error: `Row ${i}: must be an object` }
    const o = row as Record<string, unknown>
    for (let v = 1; v <= 28; v++) {
      const k = `V${v}`
      if (!(k in o)) return { ok: false, error: `Row ${i}: missing ${k}` }
      if (!Number.isFinite(Number(o[k]))) return { ok: false, error: `Row ${i}: ${k} must be finite` }
    }
    if (!Number.isFinite(Number(o.Amount)) || Number(o.Amount) < 0)
      return { ok: false, error: `Row ${i}: Amount invalid` }
    if (!Number.isFinite(Number(o.Time)) || Number(o.Time) < 0)
      return { ok: false, error: `Row ${i}: Time invalid` }
  }
  return { ok: true, data: txs as TransactionInput[] }
}

function parseCsv(text: string): { ok: true; data: TransactionInput[] } | { ok: false; error: string } {
  const lines = text.trim().split(/\r?\n/).filter(Boolean)
  if (lines.length < 2) return { ok: false, error: 'CSV needs a header row and at least one data row' }
  const header = lines[0].split(',').map((s) => s.trim())
  const expected = [...Array.from({ length: 28 }, (_, i) => `V${i + 1}`), 'Amount', 'Time']
  for (const h of expected) {
    if (!header.includes(h)) return { ok: false, error: `CSV header must include column: ${h}` }
  }
  const idx: Record<string, number> = {}
  header.forEach((h, i) => {
    idx[h] = i
  })
  const rows: TransactionInput[] = []
  for (let r = 1; r < lines.length; r++) {
    const cells = lines[r].split(',').map((s) => s.trim())
    if (cells.length < header.length) return { ok: false, error: `Row ${r}: not enough columns` }
    const o: Record<string, number> = {} as Record<string, number>
    for (const h of expected) {
      const n = Number(cells[idx[h]])
      if (!Number.isFinite(n)) return { ok: false, error: `Row ${r}: ${h} is not a number` }
      o[h] = n
    }
    rows.push(o as TransactionInput)
  }
  if (rows.length > 1000) return { ok: false, error: 'Maximum 1000 rows' }
  return { ok: true, data: rows }
}

export function BatchPanel() {

  const [mode, setMode] = useState<'json' | 'csv'>('json')
  const [text, setText] = useState('')
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [batch, setBatch] = useState<BatchPredictResponse | null>(null)
  const [flaggedOnly, setFlaggedOnly] = useState(false)
  const [compact, setCompact] = useState(false)
  const [expanded, setExpanded] = useState<Record<number, boolean>>({})
  const [sort, setSort] = useState<'idx' | 'score' | 'fraud'>('idx')

  const filteredSorted = useMemo(() => {
    if (!batch) return []
    let rows = [...batch.results]
    if (flaggedOnly) rows = rows.filter((r) => r.is_fraud)
    if (sort === 'score') rows.sort((a, b) => b.risk_score - a.risk_score)
    else if (sort === 'fraud') rows.sort((a, b) => Number(b.is_fraud) - Number(a.is_fraud))
    else rows.sort((a, b) => (a.index ?? 0) - (b.index ?? 0))
    return rows
  }, [batch, flaggedOnly, sort])

  async function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (!f) return
    setErr(null)
    try {
      const t = await f.text()
      setText(t)
      setMode('csv')
    } catch {
      setErr('Could not read file')
    }
    e.target.value = ''
  }

  async function runBatch() {
    setErr(null)
    setBatch(null)
    const parsed =
      mode === 'json'
        ? parseBatchJson(text)
        : text.trim().startsWith('{')
          ? parseBatchJson(text)
          : parseCsv(text)
    if (!parsed.ok) {
      setErr(parsed.error)
      return
    }
    setLoading(true)
    try {
      const res = await postPredictBatch(parsed.data)
      setBatch(res)
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'Batch failed')
    } finally {
      setLoading(false)
    }
  }

  const colSpan = compact ? 5 : 6

  return (
    <section className="panel">
      <h2 className="panel__title">Batch review</h2>
      <p className="panel__lead">
        Send up to 1000 rows to <code>POST /predict/batch</code> using JSON (<code>transactions</code> wrapper) or a
        CSV with columns <code>V1</code>…<code>V28</code>, <code>Amount</code>, <code>Time</code>.
      </p>
      <div className="row gap wrap batch-mode">
        <label className="toggle">
          <input type="radio" name="batchmode" checked={mode === 'json'} onChange={() => setMode('json')} />
          <span>JSON</span>
        </label>
        <label className="toggle">
          <input type="radio" name="batchmode" checked={mode === 'csv'} onChange={() => setMode('csv')} />
          <span>CSV</span>
        </label>
        <label className="btn btn--secondary file-btn">
          Upload CSV
          <input type="file" accept=".csv,text/csv" className="file-btn__input" onChange={onFile} />
        </label>
      </div>
      <textarea
        className="input input--textarea"
        rows={8}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={
          mode === 'json'
            ? '{ "transactions": [ { "V1": 0, "V2": 0, ... "Amount": 1, "Time": 0 } ] }'
            : 'V1,V2,...,Amount,Time\n0,0,...,10,0'
        }
        spellCheck={false}
        aria-label="Batch payload"
      />
      {err && <p className="panel__error">{err}</p>}
      <button type="button" className="btn btn--primary" onClick={runBatch} disabled={loading || !text.trim()}>
        {loading ? 'Running batch…' : 'Run batch'}
      </button>

      {batch && (
        <div className="batch-summary">
          <div className="chips">
            <span className="chip">{batch.n_transactions} scored</span>
            <span className="chip chip--alert">{batch.n_flagged} flagged</span>
            <span className="chip">Flag rate {(100 * batch.flag_rate).toFixed(2)}%</span>
            <span className="chip">{batch.total_latency_ms.toFixed(1)} ms total</span>
            <span className="chip">{batch.avg_latency_ms.toFixed(3)} ms / txn</span>
          </div>
          <div className="row gap wrap">
            <label className="toggle toggle--switch">
              <input type="checkbox" checked={flaggedOnly} onChange={(e) => setFlaggedOnly(e.target.checked)} />
              <span>Flagged only</span>
            </label>
            <label className="toggle toggle--switch">
              <input type="checkbox" checked={compact} onChange={(e) => setCompact(e.target.checked)} />
              <span>Compact rows</span>
            </label>
            <label className="toggle">
              <span className="panel__muted">Sort</span>
              <select value={sort} onChange={(e) => setSort(e.target.value as typeof sort)} aria-label="Sort rows">
                <option value="idx">Row order</option>
                <option value="score">Risk score</option>
                <option value="fraud">Fraud first</option>
              </select>
            </label>
          </div>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Fraud</th>
                  {!compact && <th>Score</th>}
                  <th>Risk %</th>
                  <th>Amt</th>
                  <th>Conf</th>
                </tr>
              </thead>
              <tbody>
                {filteredSorted.map((r) => {
                  const i = r.index ?? 0
                  const exp = expanded[i]
                  return (
                    <Fragment key={i}>
                      <tr
                        className={r.is_fraud ? 'data-table__row--alert' : undefined}
                        onClick={() => setExpanded((m) => ({ ...m, [i]: !m[i] }))}
                        tabIndex={0}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault()
                            setExpanded((m) => ({ ...m, [i]: !m[i] }))
                          }
                        }}
                      >
                        <td>{i}</td>
                        <td>{r.is_fraud ? 'Yes' : 'No'}</td>
                        {!compact && <td className="mono">{r.risk_score.toFixed(3)}</td>}
                        <td className="mono">{(r.risk_score_norm * 100).toFixed(1)}</td>
                        <td className="mono">{r.amount.toFixed(2)}</td>
                        <td>
                          <span className={`badge badge--${r.confidence}`}>{r.confidence}</span>
                        </td>
                      </tr>
                      {exp && (
                        <tr className="data-table__explain-row">
                          <td colSpan={colSpan} className="data-table__explain">
                            {r.explanation}
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="panel__muted">Click a row to show or hide the explanation string.</p>
        </div>
      )}
    </section>
  )
}
