import { useEffect, useMemo, useState } from 'react'
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { api, type RocCurve, type Table } from '../api'
import { Badge, Card, ErrorBox, PageHeader, Spinner, Tabs } from '../components/ui'
import DataTable from '../components/DataTable'

type Protocol = 'A' | 'B' | 'C'

const PROTOCOLS: { id: Protocol; label: string; blurb: string }[] = [
  { id: 'A', label: 'Protocol A', blurb: '60/40 stratified hold-out, random_state=42' },
  { id: 'B', label: 'Protocol B', blurb: '70/30 stratified hold-out, random_state=42 — the paper’s main setting' },
  { id: 'C', label: 'Protocol C', blurb: '10-fold stratified CV, mean ± std over folds' },
]

const ROC_COLORS = ['#818cf8', '#22d3ee', '#fb7185', '#34d399', '#fbbf24', '#c084fc']

/** ROC curve points → recharts scatter polyline ({fpr, tpr} per model). */
function toRocSeries(curves: RocCurve[]) {
  const n = Math.max(...curves.map((c) => c.points.length))
  return Array.from({ length: n }, (_, i) => {
    const row: Record<string, number | null> = { idx: i }
    for (const c of curves) row[c.model] = c.points[i]?.[1] ?? null
    return row
  })
}

export default function Benchmarks() {
  const [protocol, setProtocol] = useState<Protocol>('B')
  const [table, setTable] = useState<Table | null>(null)
  const [roc, setRoc] = useState<RocCurve[]>([])
  const [confusion, setConfusion] = useState<Table | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancel = false
    setLoading(true)
    setError(null)
    Promise.all([api.benchmarks(protocol), api.roc(protocol)])
      .then(([t, r]) => {
        if (cancel) return
        setTable(t)
        setRoc(r.curves)
      })
      .catch((e) => !cancel && setError(String(e)))
      .finally(() => !cancel && setLoading(false))
    return () => { cancel = true }
  }, [protocol])

  useEffect(() => {
    api.confusion('B').then(setConfusion).catch(() => setConfusion(null))
  }, [])

  const rocSeries = useMemo(() => (roc.length ? toRocSeries(roc) : []), [roc])

  const blurb = PROTOCOLS.find((p) => p.id === protocol)!.blurb

  return (
    <div>
      <PageHeader
        eyebrow="Model comparison"
        title="Benchmarks"
        description="How the six pipelines score under three evaluation protocols — held-out metrics, ROC curves and the confusion matrix."
      />

      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <Tabs options={PROTOCOLS.map((p) => ({ id: p.id, label: p.label }))} value={protocol} onChange={setProtocol} />
        <Badge tone="indigo">{blurb}</Badge>
      </div>

      {error && <ErrorBox error={error} />}
      {loading && <Spinner label="Loading benchmarks…" />}

      {table && !loading && (
        <div className="space-y-6">
          <Card title={`Test metrics — ${protocol}`} subtitle="Accuracy / precision / recall / F1 / AUC on the evaluation set; disease is the positive class">
            <DataTable table={table} keyName="model" dense />
          </Card>

          <div className="grid gap-6 lg:grid-cols-2">
            <Card title="ROC curves" subtitle={`True vs false positive rate — protocol ${protocol}`}>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={rocSeries} margin={{ top: 5, right: 10, left: -18, bottom: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="idx" hide />
                  <YAxis
                    domain={[0, 1]}
                    tick={{ fill: '#64748b', fontSize: 11 }}
                    tickFormatter={(v) => String(v)}
                  />
                  <Tooltip
                    contentStyle={{ background: '#0d1220', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, fontSize: 12 }}
                    labelFormatter={() => ''}
                    formatter={(v: number, name) => [v.toFixed(3), name]}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {roc.map((c, i) => (
                    <Line
                      key={c.model}
                      type="monotone"
                      dataKey={c.model}
                      stroke={ROC_COLORS[i % ROC_COLORS.length]}
                      strokeWidth={2}
                      dot={false}
                      connectNulls
                      name={`${c.model} (AUC ${c.auc.toFixed(3)})`}
                    />
                  ))}
                  {/* reference diagonal */}
                  <Line
                    type="monotone"
                    dataKey={() => null}
                    strokeDasharray="4 4"
                    stroke="rgba(255,255,255,0.2)"
                    dot={false}
                    name="chance"
                  />
                </LineChart>
              </ResponsiveContainer>
            </Card>

            <Card
              title="Held-out confusion matrices (protocol B)"
              subtitle="Rows = true class, cols = predicted · from results/18_*.csv"
            >
              {confusion ? <DataTable table={confusion} keyName="model" dense /> : <Spinner label="No confusion table" />}
            </Card>
          </div>
        </div>
      )}
    </div>
  )
}
