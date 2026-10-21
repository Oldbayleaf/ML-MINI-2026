import { useEffect, useState } from 'react'
import { Scale } from 'lucide-react'
import { api, type Table } from '../api'
import { Badge, Card, ErrorBox, PageHeader, Spinner, Tabs } from '../components/ui'
import DataTable from '../components/DataTable'

type AuditTab =
  | 'claims' | 'comparison' | 'comparison7' | 'f1-check' | 'table6'
  | 'table3-vs-table6' | 'table3-vs-table6-prf' | 'splits' | 'rankings' | 'train-vs-test'

const TABS: { id: AuditTab; label: string; title: string; subtitle: string }[] = [
  { id: 'claims', label: 'Claims ledger', title: 'Claims ledger', subtitle: 'Every claim the paper makes, with the evidence that supports or contradicts it' },
  { id: 'comparison', label: 'Ours vs paper (13)', title: 'Paper vs our replication — all 13 features', subtitle: 'Published accuracy vs what the same model scores on the same data here' },
  { id: 'comparison7', label: 'Ours vs paper (7)', title: 'Paper vs our replication — paper’s 7 features', subtitle: 'Same comparison restricted to the 7-feature subset the paper highlights' },
  { id: 'f1-check', label: 'F1 consistency', title: 'Paper F1 self-consistency', subtitle: 'Do the paper’s own precision/recall numbers reproduce its reported F1?' },
  { id: 'table6', label: 'Table 6 recomputed', title: 'Paper Table 6, recomputed', subtitle: 'The paper’s per-fold table rebuilt from the raw Cleveland data' },
  { id: 'table3-vs-table6', label: 'T3 vs T6 acc', title: 'Table 3 vs Table 6 — accuracy', subtitle: 'Internal consistency: two tables in the same paper reporting the same models' },
  { id: 'table3-vs-table6-prf', label: 'T3 vs T6 P/R/F', title: 'Table 3 vs Table 6 — precision/recall/F1', subtitle: 'Same cross-check on the other three metrics' },
  { id: 'splits', label: 'Split sizes', title: 'Split sizes', subtitle: 'Exact train/test row counts per protocol — n=303 stratified' },
  { id: 'rankings', label: 'Model rankings', title: 'Model rankings', subtitle: 'How models rank in the paper vs in our replication — does the leaderboard survive?' },
  { id: 'train-vs-test', label: 'Train vs test', title: 'Train vs test accuracy', subtitle: 'The overfitting gap: training scores are always flattering, held-out scores are the truth' },
]

export default function Audit() {
  const [tab, setTab] = useState<AuditTab>('claims')
  const [table, setTable] = useState<Table | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancel = false
    setLoading(true)
    setError(null)
    api
      .audit(tab)
      .then((t) => !cancel && setTable(t))
      .catch((e) => !cancel && setError(String(e)))
      .finally(() => !cancel && setLoading(false))
    return () => { cancel = true }
  }, [tab])

  const spec = TABS.find((t) => t.id === tab)!

  return (
    <div>
      <PageHeader
        eyebrow="Tab 3 · Replication audit"
        title="Paper vs. Replication"
        description="An independent re-run of the published study on the exact same Cleveland data. Where our numbers agree, the claim is supported; where they diverge, the discrepancy is quantified rather than glossed over."
      />

      <div className="mb-5 overflow-x-auto pb-1">
        <Tabs options={TABS.map((t) => ({ id: t.id, label: t.label }))} value={tab} onChange={setTab} />
      </div>

      {error && <ErrorBox error={error} />}
      {loading && <Spinner label="Loading audit table…" />}

      {table && !loading && (
        <Card
          title={<span className="flex items-center gap-2"><Scale className="h-3.5 w-3.5 text-accent-indigo" />{spec.title}</span>}
          subtitle={spec.subtitle}
          right={<Badge tone="cyan">results/*.csv</Badge>}
        >
          <DataTable table={table} />
          <p className="mt-3 text-[11px] leading-relaxed text-slate-500">
            Tables are served verbatim from the committed replication outputs — nothing here is
            recomputed in the browser, so the dashboard and the published numbers can never drift
            apart. Regenerate with <code className="num">python run_replication.py</code>.
          </p>
        </Card>
      )}
    </div>
  )
}
