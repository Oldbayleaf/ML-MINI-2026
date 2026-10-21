import { useEffect, useMemo, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts'
import { api, pct, toRows, type IsolationRecord, type Table } from '../api'
import { Card, ErrorBox, PageHeader, Spinner } from '../components/ui'
import DataTable from '../components/DataTable'

const ACC_KEYS = [
  { key: 'acc_A', name: 'Protocol A', color: '#818cf8' },
  { key: 'acc_B', name: 'Protocol B', color: '#22d3ee' },
  { key: 'acc_C', name: 'Protocol C (CV mean)', color: '#34d399' },
]

const tick = { fill: '#64748b', fontSize: 11 }
const tooltipStyle = { background: '#0d1220', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, fontSize: 12 }

/** Wide table (label + acc_A/B/C) → rows for a grouped bar chart. */
function wideRows(t: Table, labelKey: string) {
  return toRows(t).map((r) => {
    const out: Record<string, string | number> = { label: String(r[labelKey] ?? '') }
    for (const k of ACC_KEYS) out[k.name] = Number(r[k.key])
    return out
  })
}

export default function Lab() {
  const [knn, setKnn] = useState<Table | null>(null)
  const [dtSweep, setDtSweep] = useState<Table | null>(null)
  const [dtCrit, setDtCrit] = useState<Table | null>(null)
  const [svm, setSvm] = useState<Table | null>(null)
  const [rf, setRf] = useState<Table | null>(null)
  const [rfImp, setRfImp] = useState<Table | null>(null)
  const [featScores, setFeatScores] = useState<Table | null>(null)
  const [logreg, setLogreg] = useState<Table | null>(null)
  const [iso, setIso] = useState<IsolationRecord[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const load = <T,>(p: Promise<T>, set: (v: T) => void) => p.then(set).catch((e) => setError(String(e)))
    load(api.lab('knn-sweep'), setKnn)
    load(api.lab('dt-sweep'), setDtSweep)
    load(api.lab('dt-criterion'), setDtCrit)
    load(api.lab('svm-kernels'), setSvm)
    load(api.lab('rf-comparison'), setRf)
    load(api.lab('rf-importance'), setRfImp)
    load(api.lab('feature-scores'), setFeatScores)
    load(api.lab('logreg-coef'), setLogreg)
    load(api.isolation().then((r) => r.records), setIso)
  }, [])

  // KNN: one point per k, one line per distance metric (protocol-B accuracy).
  // CSV shape: metric,k,acc_A,acc_B,acc_C,acc_C_std,innerCV_trainB
  const knnPoints = useMemo(() => {
    if (!knn) return { rows: [] as Record<string, string | number>[], metrics: [] as string[] }
    const raw = toRows(knn)
    const metrics = [...new Set(raw.map((r) => String(r.metric)))]
    const byK = new Map<number, Record<string, string | number>>()
    for (const r of raw) {
      const k = Number(r.k)
      const row = byK.get(k) ?? { k }
      row[String(r.metric)] = Number(r.acc_B)
      byK.set(k, row)
    }
    return { rows: [...byK.values()].sort((a, b) => Number(a.k) - Number(b.k)), metrics }
  }, [knn])

  const dtPoints = useMemo(() => {
    if (!dtSweep) return { rows: [] as Record<string, string | number>[], criteria: [] as string[] }
    const raw = toRows(dtSweep)
    const criteria = [...new Set(raw.map((r) => String(r.criterion)))]
    const byK = new Map<number, Record<string, string | number>>()
    for (const r of raw) {
      const k = Number(r.k_features)
      const row = byK.get(k) ?? { k }
      row[String(r.criterion)] = Number(r.acc_B)
      byK.set(k, row)
    }
    return { rows: [...byK.values()].sort((a, b) => Number(a.k) - Number(b.k)), criteria }
  }, [dtSweep])

  const rfImpRows = useMemo(() => {
    if (!rfImp) return []
    const first = rfImp.columns[0]
    return toRows(rfImp)
      .map((r) => ({ feature: String(r[first]), importance: Number(r.importance) }))
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 13)
  }, [rfImp])

  const logregRows = useMemo(() => {
    if (!logreg) return []
    const first = logreg.columns[0]
    return toRows(logreg)
      .map((r) => ({ feature: String(r[first]), coef: Number(r.coef) }))
      .sort((a, b) => Math.abs(b.coef) - Math.abs(a.coef))
      .slice(0, 12)
  }, [logreg])

  const isoRows = useMemo(() => {
    if (!iso) return []
    return iso.filter((r) => String(r.protocol) === 'B')
  }, [iso])

  return (
    <div>
      <PageHeader
        eyebrow="Tab 4 · Hyperparameter lab"
        title="Hyperparameter & Feature Lab"
        description="Every sweep the paper ran: KNN's k, decision-tree depth via feature count, SVM kernels, Random-Forest tuning, and what the features themselves say about the target. Protocol B accuracy (70/30 hold-out) is highlighted throughout."
      />

      {error && <ErrorBox error={error} />}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* KNN k sweep */}
        <Card title="KNN — k sweep" subtitle="Test accuracy vs number of neighbours, per distance metric (protocol B)">
          {!knn && <Spinner /> }
          {knn && (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={knnPoints.rows} margin={{ top: 5, right: 10, left: -18, bottom: 0 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="k" tick={tick} label={{ value: 'k', position: 'insideBottomRight', fill: '#64748b', fontSize: 11 }} />
                <YAxis domain={[0.6, 0.95]} tick={tick} tickFormatter={(v) => pct(Number(v), 0)} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => [pct(v, 1), 'accuracy B']} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {knnPoints.metrics.map((m, i) => (
                  <Line key={m} type="monotone" dataKey={m} stroke={['#818cf8', '#22d3ee', '#fb7185'][i % 3]} strokeWidth={2} dot={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          )}
        </Card>

        {/* DT feature-count sweep */}
        <Card title="Decision Tree — feature-count sweep" subtitle="Accuracy vs number of selected features, per split criterion">
          {!dtSweep && <Spinner />}
          {dtSweep && (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={dtPoints.rows} margin={{ top: 5, right: 10, left: -18, bottom: 0 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="k" tick={tick} label={{ value: '# features', position: 'insideBottomRight', fill: '#64748b', fontSize: 11 }} />
                <YAxis domain={[0.6, 0.95]} tick={tick} tickFormatter={(v) => pct(Number(v), 0)} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => [pct(v, 1), 'accuracy B']} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {dtPoints.criteria.map((c, i) => (
                  <Line key={c} type="monotone" dataKey={c} stroke={['#fbbf24', '#c084fc'][i % 2]} strokeWidth={2} dot={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          )}
        </Card>

        {/* SVM kernels */}
        <Card title="SVM — kernel comparison" subtitle="Linear / poly / RBF / sigmoid across all three protocols">
          {!svm && <Spinner />}
          {svm && (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={wideRows(svm, svm.columns[0])} margin={{ top: 5, right: 10, left: -18, bottom: 0 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                <XAxis dataKey="label" tick={tick} />
                <YAxis domain={[0.6, 0.95]} tick={tick} tickFormatter={(v) => pct(Number(v), 0)} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => [pct(v, 1), 'accuracy']} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {ACC_KEYS.map((k) => (
                  <Bar key={k.key} dataKey={k.name} fill={k.color} radius={[4, 4, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        {/* RF default vs tuned */}
        <Card title="Random Forest — default vs tuned" subtitle="Does grid-search tuning actually beat the defaults? (5-fold CV inside the train split only)">
          {!rf && <Spinner />}
          {rf && (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={wideRows(rf, rf.columns[0])} margin={{ top: 5, right: 10, left: -18, bottom: 0 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                <XAxis dataKey="label" tick={tick} />
                <YAxis domain={[0.6, 0.95]} tick={tick} tickFormatter={(v) => pct(Number(v), 0)} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => [pct(v, 1), 'accuracy']} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {ACC_KEYS.map((k) => (
                  <Bar key={k.key} dataKey={k.name} fill={k.color} radius={[4, 4, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        {/* RF feature importance */}
        <Card title="Random Forest — feature importance" subtitle="Mean decrease in impurity, top 13">
          {!rfImp && <Spinner />}
          {rfImpRows.length > 0 && (
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={rfImpRows} layout="vertical" margin={{ top: 5, right: 16, left: 12, bottom: 0 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.06)" horizontal={false} />
                <XAxis type="number" tick={tick} />
                <YAxis type="category" dataKey="feature" width={80} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => [v.toFixed(4), 'importance']} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
                <Bar dataKey="importance" fill="#818cf8" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        {/* Logistic regression coefficients */}
        <Card title="Logistic Regression — coefficients" subtitle="Largest |coef| on the standardized features (protocol-B fit): direction and strength of effect">
          {!logreg && <Spinner />}
          {logregRows.length > 0 && (
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={logregRows} layout="vertical" margin={{ top: 5, right: 16, left: 24, bottom: 0 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.06)" horizontal={false} />
                <XAxis type="number" tick={tick} />
                <YAxis type="category" dataKey="feature" width={92} tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => [v.toFixed(3), 'coefficient']} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
                <Bar dataKey="coef" fill="#22d3ee" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>
      </div>

      <div className="mt-6 space-y-6">
        <Card title="Feature-selection scores" subtitle="Mutual information and chi² per feature, with rank stability across CV folds">
          {featScores ? <DataTable table={featScores} keyName={featScores.columns[0]} dense /> : <Spinner />}
        </Card>

        <Card title="Decision-tree criterion" subtitle="Entropy vs gini head-to-head">
          {dtCrit ? <DataTable table={dtCrit} keyName={dtCrit.columns[0]} dense /> : <Spinner />}
        </Card>

        <Card
          title="Isolation Forest — anomaly detection"
          subtitle="Accuracy after removing model-detected outliers (protocol B shown); anomaly removal helps some models and hurts others"
        >
          {iso ? (
            <DataTable
              table={{
                columns: ['model', 'outlier', 'accuracy'],
                rows: {
                  model: isoRows.map((r) => r.model),
                  outlier: isoRows.map((r) => r.outlier),
                  accuracy: isoRows.map((r) => r.accuracy),
                },
              }}
              dense
            />
          ) : <Spinner />}
        </Card>
      </div>
    </div>
  )
}
