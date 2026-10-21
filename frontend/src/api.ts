/* Typed client for the CardioScan FastAPI backend (proxied at /api by Vite). */

/** Backend tables are column-oriented: rows = { column: values[] }. */
export interface Table {
  columns: string[]
  rows: Record<string, (string | number | null)[]>
}

export interface ModelMeta {
  model: string
  held_out_accuracy: number
  ci_half: number
  n_test: number
  auc: number
}

export interface Prediction {
  model: string
  prediction: number
  probability: number
  held_out_accuracy: number
  ci_half: number
}

export interface PredictResponse {
  models: Prediction[]
  consensus: { votes_for_disease: number; total: number; mean_probability: number; label: string }
}

export interface Patient {
  age: number
  sex: number
  cp: number
  trestbps: number
  chol: number
  fbs: number
  restecg: number
  thalach: number
  exang: number
  oldpeak: number
  slope: number
  ca: number
  thal: number
}

export interface RocCurve {
  model: string
  auc: number
  points: [number, number][]
}

export interface IsolationRecord {
  model: string
  protocol: string
  outlier: string
  accuracy: number
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`${path} → HTTP ${res.status}`)
  return res.json() as Promise<T>
}

export const api = {
  health: () => get<{ status: string; models: string[]; n_rows: number }>('/api/health'),
  modelsMeta: () => get<{ models: ModelMeta[]; note: string }>('/api/models'),
  predict: async (patient: Patient): Promise<PredictResponse> => {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patient),
    })
    if (!res.ok) throw new Error(`predict → HTTP ${res.status}`)
    return res.json()
  },
  benchmarks: (p: 'A' | 'B' | 'C') => get<Table>(`/api/benchmarks/${p}`),
  confusion: (p: 'B') => get<Table>(`/api/confusion/${p}`),
  samplingNoise: () => get<Table>('/api/sampling-noise'),
  roc: (p: 'A' | 'B' | 'C') => get<{ protocol: string; curves: RocCurve[] }>(`/api/roc/${p}`),
  audit: (t: string) => get<Table>(`/api/audit/${t}`),
  stats: (t: string) => get<Table>(`/api/stats/${t}`),
  lab: (t: string) => get<Table>(`/api/lab/${t}`),
  isolation: () => get<{ records: IsolationRecord[] }>('/api/lab/isolation'),
  eda: (t: string) => get<Table>(`/api/eda/${t}`),
  classBalance: () => get<{ labels: string[]; counts: number[]; total: number }>('/api/eda/class-balance'),
  distributions: () =>
    get<
      Record<
        string,
        {
          by_target: {
            target: number
            mean: number
            std: number
            q25: number
            q50: number
            q75: number
            min: number
            max: number
            values: number[]
          }[]
        }
      >
    >('/api/eda/distributions'),
}

/** Convert a column-oriented table into an array of row objects. */
export function toRows<T = Record<string, string | number | null>>(table: Table): T[] {
  const n = table.columns.length ? (table.rows[table.columns[0]]?.length ?? 0) : 0
  const out: Record<string, string | number | null>[] = []
  for (let i = 0; i < n; i++) {
    const row: Record<string, string | number | null> = {}
    for (const c of table.columns) row[c] = table.rows[c]?.[i] ?? null
    out.push(row)
  }
  return out as T[]
}

/** 0.838 → "83.8%" ; undefined-safe. */
export const pct = (v: number | null | undefined, digits = 1) =>
  v == null || Number.isNaN(v) ? '—' : `${(v * 100).toFixed(digits)}%`

/** 0.838 ± 0.064 → "0.838 ± 0.064" with sensible fallbacks. */
export const pm = (m: number | null | undefined, s: number | null | undefined) =>
  m == null ? '—' : s == null || Number.isNaN(s) ? String(m) : `${m.toFixed(3)} ± ${Number(s).toFixed(3)}`
