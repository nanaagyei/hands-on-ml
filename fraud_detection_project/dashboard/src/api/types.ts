export type ApiErrorBody = { error: string; status: number }

export type HealthResponse =
  | { status: 'healthy'; timestamp: string }
  | { status: 'unhealthy'; reason: string }

export type ModelInfoResponse = {
  model_name: string
  n_features: number
  feature_names: string[]
  threshold: number
  threshold_label: string
  test_auc_pr: number | null
  test_recall: number | null
  test_precision: number | null
  fraud_rate_train: number | null
}

export type TransactionInput = Record<string, number> & {
  Amount: number
  Time: number
}

export type PredictResult = {
  is_fraud: boolean
  risk_score: number
  risk_score_norm: number
  threshold: number
  confidence: string
  explanation: string
  amount: number
  latency_ms: number
  timestamp: string
  index?: number
}

export type BatchPredictResponse = {
  results: PredictResult[]
  n_transactions: number
  n_flagged: number
  flag_rate: number
  total_latency_ms: number
  avg_latency_ms: number
}

export type MonitorAlert = {
  type: string
  severity: string
  message: string
  psi?: number
  rate?: number
}

export type MonitorNoData = { status: 'no_data'; message: string }

export type MonitorStats = {
  timestamp: string
  n_total: number
  n_window: number
  psi: number
  psi_status: 'ok' | 'warn' | 'alert'
  fraud_rate_window: number
  fraud_rate_total: number
  score_mean_window: number
  score_std_window: number
  score_p95_window: number
  alerts: MonitorAlert[]
}

export type MonitorStatusResponse = MonitorNoData | MonitorStats

export type MonitorResetResponse = {
  status: string
  message: string
  timestamp: string
}
