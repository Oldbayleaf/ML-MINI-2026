import { useEffect, useState } from 'react'
import {
  CartesianGrid, ErrorBar, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Tooltip, XAxis, YAxis, Bar, BarChart, Cell,
} from 'recharts'
import { api, toRows, type Table } from '../api'
import { Badge, Card, ErrorBox, PageHeader, Spinner, Stat } from '../components/ui'
import DataTable from '../components/DataTable'

/** Shared fetcher: grabs several stats tables at once, tracks loading + first error. */
function useStatsTables(names: string[]) {
  const [tables, setTables] = useState<Record<string, Table>>({})
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let cancel = false
    Promise.all(names.map(async (n) => [n, await api.stats(n)] as const))
      .then((pairs) => {
        if (cancel) return
        setTables(Object.fromEntries(pairs))
      })
      .catch((e) => !cancel && setError(String(e)))
    return () => {
      cancel = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  return { tables, error }
}

export default function Statistics() {
  const { tables, error } = useStatsTables([
    'stability', 'mcnemar', 'operating-points', 'threshold-sweep', 'permutation', 'hgb',
  ])
  const loading = Object.keys(tables).length === 0 && !error

  // ---- repeated-CV stability: mean accuracy with fold-to-fold error bars ----
  const stabilityRows = tables.stability ? toRows(tables.stability) : []
  const stabilitySeries = stabilityRows.map((r) => ({
    model: String(r.model),
    mean: Number(r.mean_accuracy),
    sd: Number(r.std_across_folds),
    rank: Number(r.rank),
  }))

  // ---- McNemar: p-value per pair, flagged significant at 0.05 ----
  const mcnemarRows = tables.mcnemar ? toRows(tables.mcnemar) : []
  const nSig = mcnemarRows.filter((r) => String(r.significant_05) === 'yes').length

  // ---- permutation importance (top model: the best by repeated CV) ----
  const permRows = tables.permutation ? toRows(tables.permutation) : []
  const permModel = 'Logistic Regression'
  const permSeries = permRows
    .filter((r) => String(r.model) === permModel)
    .map((r) => ({
      feature: String(r.feature),
      importance: Number(r.importance_mean),
      sd: Number(r.importance_std),
    }))

  // ---- threshold sweep: sensitivity/specificity vs cut-off, best model ----
  const sweepRows = tables['threshold-sweep'] ? toRows(tables['threshold-sweep']) : []
  const sweepModel = 'Logistic Regression'
  const sweepSeries = sweepRows
    .filter((r) => String(r.model) === sweepModel)
    .map((r) => ({
      threshold: Number(r.threshold),
      sensitivity: Number(r.sensitivity),
      specificity: Number(r.specificity),
      accuracy: Number(r.accuracy),
    }))

  const axis = { fill: '#64748b', fontSize: 11 }
  const grid = 'rgba(255,255,255,0.06)'
  const tip = {
    contentStyle: {
      background: '#0d1220', border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: 12, fontSize: 12,
    },
  }

  return (
    <div>
      <PageHeader
        eyebrow="Statistical extensions"
        title="Statistical Lens"
        description="The published replication reports point estimates. This page asks whether those gaps are real: repeated cross-validation for stability, exact McNemar tests for pairwise superiority, permutation importance for model-agnostic feature reliance, and the sensitivity/specificity trade-off behind the accuracy headline."
      />

      {error && <ErrorBox error={error} />}
      {loading && !error && <Spinner label="Loading statistical tables…" />}

      {!loading && (
        <>
          {/* ---------------------------------------------- headline stats */}
          <div className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Stat
              label="Repeated-CV winner"
              value={String(stabilitySeries[0]?.model ?? '—')}
              hint={`${(stabilitySeries[0]?.mean ?? 0).toFixed(3)} mean accuracy over 10×10 folds`}
              tone="good"
            />
            <Stat
              label="Significant pairs"
              value={`${nSig} / ${mcnemarRows.length}`}
              hint="McNemar exact p < 0.05 on the held-out split"
              tone={nSig > 0 ? 'warn' : 'bad'}
            />
            <Stat
              label="Test-set size"
              value="n = 91"
              hint="One patient ≈ 1.1 accuracy points; 95% CI ≈ ±7 pts"
            />
            <Stat
              label="HGB vs paper's models"
              value="not better"
              hint="Modern boosting does not beat the best paper model on 303 rows"
              tone="warn"
            />
          </div>

          {/* ---------------------------------------------- stability chart */}
          <Card
            title="Repeated cross-validation stability"
            subtitle="Mean accuracy over 10×10 stratified folds; error bars = fold-to-fold std. Wider bars mean the single-split ranking is fragile."
          >
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={stabilitySeries} margin={{ top: 8, right: 16, left: -8, bottom: 0 }}>
                <CartesianGrid stroke={grid} vertical={false} />
                <XAxis dataKey="model" tick={axis} interval={0} angle={-18} textAnchor="end" height={70} />
                <YAxis domain={[0.7, 0.9]} tick={axis} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                <ReferenceLine y={0.85} stroke="rgba(255,255,255,0.12)" strokeDasharray="4 4" />
                <Tooltip {...tip} formatter={(v: number) => `${(v * 100).toFixed(1)}%`} />
                <Line
                  type="monotone"
                  dataKey="mean"
                  stroke="#818cf8"
                  strokeWidth={2}
                  dot={{ r: 3, fill: '#818cf8' }}
                >
                  <ErrorBar dataKey="sd" width={10} strokeWidth={1} stroke="#c7d2fe" />
                </Line>
              </LineChart>
            </ResponsiveContainer>
            <p className="mt-3 text-xs text-slate-400">
              The top four models sit inside each other&apos;s error bars — on 303 patients the
              leaderboard order is not statistically meaningful. Only the Decision Tree separates.
            </p>
          </Card>

          {/* ---------------------------------------------- McNemar + permutation */}
          <div className="mt-6 grid gap-6 lg:grid-cols-2">
            <Card
              title="Pairwise McNemar tests"
              subtitle="Exact test on discordant predictions (held-out split). A significant result means one model genuinely beats the other, not just scores higher."
              right={<Badge tone={nSig > 0 ? 'warn' : 'slate'}>{nSig} significant</Badge>}
            >
              <DataTable table={tables.mcnemar!} keyName="model_a" dense />
              <p className="mt-3 text-xs text-slate-400">
                Most pairs are non-significant: with 91 test patients the discordant count is too
                small to separate 0.84 from 0.87. This is why the replication refuses to crown a
                single winner.
              </p>
            </Card>

            <Card
              title={`Permutation importance — ${permModel}`}
              subtitle="Model-agnostic: how much does held-out accuracy drop when this feature is shuffled? Bars are mean ± std over 30 repeats."
            >
              <ResponsiveContainer width="100%" height={340}>
                <BarChart
                  data={permSeries}
                  layout="vertical"
                  margin={{ top: 8, right: 24, left: 16, bottom: 0 }}
                >
                  <CartesianGrid stroke={grid} horizontal={false} />
                  <XAxis type="number" tick={axis} tickFormatter={(v) => v.toFixed(2)} />
                  <YAxis type="category" dataKey="feature" width={72} tick={axis} />
                  <ReferenceLine x={0} stroke="rgba(255,255,255,0.15)" />
                  <Tooltip {...tip} formatter={(v: number) => v.toFixed(3)} />
                  <Bar dataKey="importance" radius={[0, 3, 3, 0]}>
                    {permSeries.map((r) => (
                      <Cell key={r.feature} fill={r.importance >= 0 ? '#818cf8' : '#fb7185'} />
                    ))}
                  </Bar>
                  <ErrorBar dataKey="sd" width={8} strokeWidth={1} stroke="#c7d2fe" />
                </BarChart>
              </ResponsiveContainer>
              <p className="mt-2 text-xs text-slate-400">
                A near-zero bar means the fitted model barely relies on that feature; a negative
                bar means shuffling it actually helped — noise, or a sign the feature was hurting.
              </p>
            </Card>
          </div>

          {/* ---------------------------------------------- threshold trade-off */}
          <Card
            className="mt-6"
            title={`Decision threshold trade-off — ${sweepModel}`}
            subtitle="Calibrated probability cut-off on the held-out split. Lowering the threshold catches more disease but flags more healthy patients as false alarms."
          >
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={sweepSeries} margin={{ top: 8, right: 16, left: -8, bottom: 0 }}>
                <CartesianGrid stroke={grid} vertical={false} />
                <XAxis
                  dataKey="threshold"
                  type="number"
                  domain={[0.05, 0.95]}
                  tick={axis}
                  tickFormatter={(v) => v.toFixed(2)}
                />
                <YAxis domain={[0, 1]} tick={axis} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                <Tooltip {...tip} formatter={(v: number) => `${(v * 100).toFixed(1)}%`} />
                <Line type="monotone" dataKey="sensitivity" stroke="#22d3ee" strokeWidth={2} dot={false} name="Sensitivity (recall)" />
                <Line type="monotone" dataKey="specificity" stroke="#fb7185" strokeWidth={2} dot={false} name="Specificity" />
                <Line type="monotone" dataKey="accuracy" stroke="#818cf8" strokeWidth={2} strokeDasharray="5 4" dot={false} name="Accuracy" />
                <ReferenceLine x={0.5} stroke="rgba(255,255,255,0.2)" strokeDasharray="3 3" />
              </LineChart>
            </ResponsiveContainer>
            <p className="mt-3 text-xs text-slate-400">
              The dashed vertical line is the default 0.5 cut-off. Accuracy peaks near it, but a
              screening tool would slide left to raise sensitivity at the cost of more false alarms.
            </p>
          </Card>

          {/* ---------------------------------------------- full operating points */}
          <Card
            className="mt-6"
            title="Clinical operating points"
            subtitle="Model behaviour at the default 0.5 cut-off and at a sensitivity-first cut-off. NPV is undefined when no patient is predicted healthy."
          >
            <DataTable table={tables['operating-points']!} keyName="model" dense />
          </Card>

          {/* ---------------------------------------------- HGB yardstick */}
          <Card
            className="mt-6"
            title="HistGradientBoosting yardstick"
            subtitle="A modern gradient-boosting baseline the paper never ran, evaluated under the exact same three protocols."
          >
            <DataTable table={tables.hgb!} keyName="protocol" dense />
            <p className="mt-3 text-xs text-slate-400">
              Boosting does not rescue the task: on 303 patients it lands in the same band as the
              paper&apos;s linear models. The ceiling here is the dataset size, not the model family.
            </p>
          </Card>
        </>
      )}
    </div>
  )
}
