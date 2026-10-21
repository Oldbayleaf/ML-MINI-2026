import { useEffect, useMemo, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { api, pct, toRows, type Table } from '../api'
import { Card, ErrorBox, PageHeader, Spinner } from '../components/ui'
import DataTable from '../components/DataTable'

const tick = { fill: '#64748b', fontSize: 11 }
const tooltipStyle = { background: '#0d1220', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, fontSize: 12 }
const FEATURE_LABELS: Record<string, string> = {
  age: 'Age', sex: 'Sex', cp: 'Chest pain type', trestbps: 'Resting BP', chol: 'Cholesterol',
  fbs: 'Fasting glucose > 120', restecg: 'Resting ECG', thalach: 'Max HR', exang: 'Exercise angina',
  oldpeak: 'ST depression', slope: 'ST slope', ca: 'Major vessels', thal: 'Thalassemia',
}
type Dist = Awaited<ReturnType<typeof api.distributions>>
type GroupStats = Dist[string]['by_target'][number]

export default function EDA() {
  const [describe, setDescribe] = useState<Table | null>(null)
  const [corr, setCorr] = useState<Table | null>(null)
  const [balance, setBalance] = useState<{ labels: string[]; counts: number[]; total: number } | null>(null)
  const [dists, setDists] = useState<Dist | null>(null)
  const [rows, setRows] = useState<Table | null>(null)
  const [selected, setSelected] = useState<string>('age')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const load = <T,>(p: Promise<T>, set: (v: T) => void) => p.then(set).catch((e) => setError(String(e)))
    load(api.eda('describe'), setDescribe)
    load(api.eda('correlations'), setCorr)
    load(api.classBalance(), setBalance)
    load(api.distributions(), setDists)
    load(api.eda('rows'), setRows)
  }, [])

  // horizontal bar: signed correlation per feature (CSV: <feature>,corr_with_target)
  const corrBar = useMemo(() => {
    if (!corr) return []
    const first = corr.columns[0]
    return toRows(corr)
      .map((r) => ({ feature: String(r[first]), value: Number(r['corr_with_target']) }))
      .filter((r) => Number.isFinite(r.value))
      .sort((a, b) => b.value - a.value)
  }, [corr])

  // distribution of the selected feature: box-ish stats + jittered raw values
  const groups: GroupStats[] = dists?.[selected]?.by_target ?? []
  const featureOptions = dists ? Object.keys(dists) : []

  return (
    <div>
      <PageHeader
        eyebrow="Tab 5 · Dataset"
        title="Exploratory Data Analysis"
        description="303 patients, 13 features, 45.8% with disease. Six missing values live in ca and thal — imputed inside each pipeline's training fold, never globally."
      />

      {error && <ErrorBox error={error} />}

      <div className="mb-6 grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
        {/* class balance donut */}
        <Card title="Class balance" subtitle="Disease vs healthy in the Cleveland set">
          {!balance && <Spinner />}
          {balance && (
            <>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie
                    data={balance.labels.map((label, i) => ({ name: label, value: balance.counts[i] }))}
                    dataKey="value"
                    nameKey="name"
                    innerRadius={55}
                    outerRadius={85}
                    paddingAngle={3}
                    stroke="none"
                  >
                    {balance.labels.map((_, i) => (
                      <Cell key={i} fill={i === 0 ? '#34d399' : '#fb7185'} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={tooltipStyle} formatter={(v: number, n) => [`${v} (${pct(v / balance.total, 1)})`, n]} />
                </PieChart>
              </ResponsiveContainer>
              <p className="text-center text-xs text-slate-400">
                <span className="num">{balance.counts[0]}</span> healthy ·{' '}
                <span className="num">{balance.counts[1]}</span> disease · total{' '}
                <span className="num">{balance.total}</span>
              </p>
            </>
          )}
        </Card>

        {/* correlation with target */}
        <Card title="Correlation with disease target" subtitle="Pearson r per feature (point-biserial for binary flags) — sorted strong → weak">
          {!corr && <Spinner />}
          {corrBar.length > 0 && (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={corrBar} margin={{ top: 5, right: 10, left: -24, bottom: 40 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                <XAxis dataKey="feature" tick={{ ...tick, fontSize: 10 }} angle={-35} textAnchor="end" interval={0} />
                <YAxis tick={tick} domain={['auto', 'auto']} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => [v.toFixed(3), 'corr']} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {corrBar.map((r) => (
                    <Cell key={r.feature} fill={r.value >= 0 ? '#fb7185' : '#34d399'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>
      </div>

      {/* feature distribution explorer */}
      <Card
        title="Distribution by outcome"
        subtitle="Pick a feature to see its spread in healthy vs diseased patients (values from heart_clean.csv — imputed for EDA only)"
        right={
          <select
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
            className="rounded-lg border border-white/10 bg-base-850 px-2.5 py-1.5 text-xs text-slate-200 outline-none focus:border-accent-indigo/40"
          >
            {featureOptions.map((f) => (
              <option key={f} value={f} className="bg-base-850">{FEATURE_LABELS[f] ?? f}</option>
            ))}
          </select>
        }
      >
        {groups.length === 0 && <Spinner />}
        {groups.length > 0 && (
          <div className="grid gap-6 md:grid-cols-2">
            {groups.map((g) => (
              <div key={g.target} className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4">
                <div className="mb-3 flex items-center justify-between">
                  <p className={`text-sm font-medium ${g.target === 1 ? 'text-rose-300' : 'text-emerald-300'}`}>
                    {g.target === 1 ? 'Disease' : 'Healthy'}
                  </p>
                  <p className="num text-[11px] text-slate-500">n = {g.values.length}</p>
                </div>
                <div className="grid grid-cols-4 gap-2 text-center">
                  {[
                    ['median', g.q50], ['mean', g.mean], ['q25', g.q25], ['q75', g.q75],
                  ].map(([label, v]) => (
                    <div key={String(label)} className="rounded-lg bg-white/[0.03] py-2">
                      <p className="label-mono text-[9px]">{String(label)}</p>
                      <p className="num mt-0.5 text-sm text-slate-200">{Number(v).toFixed(1)}</p>
                    </div>
                  ))}
                </div>
                {/* strip plot of raw values */}
                <div className="relative mt-3 h-12 rounded-lg bg-white/[0.02]">
                  {(() => {
                    const lo = Math.min(...groups.flatMap((x) => x.values))
                    const hi = Math.max(...groups.flatMap((x) => x.values))
                    const span = hi - lo || 1
                    return g.values.map((v, i) => (
                      <span
                        key={i}
                        className={`absolute top-1/2 h-3 w-[2px] -translate-y-1/2 rounded-full ${
                          g.target === 1 ? 'bg-accent-rose/60' : 'bg-accent-emerald/60'
                        }`}
                        style={{ left: `${((v - lo) / span) * 96 + 2}%` }}
                      />
                    ))
                  })()}
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

      <div className="mt-6 space-y-6">
        <Card title="Summary statistics" subtitle="Per-feature mean / std / quartiles over all 303 patients">
          {describe ? <DataTable table={describe} keyName={describe.columns[0]} dense /> : <Spinner />}
        </Card>
        <Card title="Raw cohort" subtitle="All 303 rows with the binary target — exactly what the pipelines see after encoding">
          {rows ? <DataTable table={rows} keyName="row" dense maxRows={25} /> : <Spinner />}
          {rows && (
            <p className="mt-2 text-[11px] text-slate-500">
              Showing the first 25 of 303 rows; the API serves all of them.
            </p>
          )}
        </Card>
      </div>
    </div>
  )
}
