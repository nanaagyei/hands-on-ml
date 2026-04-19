import { apiUrl } from '../config'
import type {
  ApiErrorBody,
  BatchPredictResponse,
  HealthResponse,
  ModelInfoResponse,
  MonitorResetResponse,
  MonitorStatusResponse,
  PredictResult,
  TransactionInput,
} from './types'

async function parseJson<T>(res: Response): Promise<T> {
  const text = await res.text()
  let data: unknown
  try {
    data = text ? JSON.parse(text) : {}
  } catch {
    throw new Error(`Invalid JSON (${res.status})`)
  }
  if (!res.ok) {
    const err = data as Partial<ApiErrorBody>
    throw new Error(err.error || `Request failed (${res.status})`)
  }
  return data as T
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(apiUrl('/health'))
  let data: HealthResponse
  try {
    data = (await res.json()) as HealthResponse
  } catch {
    throw new Error('Health response was not valid JSON')
  }
  return data
}

export async function getModelInfo(): Promise<ModelInfoResponse> {
  const res = await fetch(apiUrl('/model/info'))
  return parseJson<ModelInfoResponse>(res)
}

export async function postPredict(
  body: TransactionInput,
): Promise<PredictResult> {
  const res = await fetch(apiUrl('/predict'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  return parseJson<PredictResult>(res)
}

export async function postPredictBatch(transactions: TransactionInput[]) {
  const res = await fetch(apiUrl('/predict/batch'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ transactions }),
  })
  return parseJson<BatchPredictResponse>(res)
}

export async function getMonitorStatus(): Promise<MonitorStatusResponse> {
  const res = await fetch(apiUrl('/monitor/status'))
  return parseJson<MonitorStatusResponse>(res)
}

export async function postMonitorReset(): Promise<MonitorResetResponse> {
  const res = await fetch(apiUrl('/monitor/reset'), { method: 'POST' })
  return parseJson<MonitorResetResponse>(res)
}
