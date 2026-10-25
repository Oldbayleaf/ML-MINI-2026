import { useEffect, useState } from 'react'
import { Activity, ShieldCheck, Stethoscope } from 'lucide-react'
import { api } from './api'
import Diagnosis from './pages/Diagnosis'
import Benchmarks from './pages/Benchmarks'
import Audit from './pages/Audit'

type PageId = 'diagnosis' | 'benchmarks' | 'audit'

const NAV: { id: PageId; label: string; icon: typeof Stethoscope; caption: string }[] = [
  { id: 'diagnosis', label: 'Diagnosis', icon: Stethoscope, caption: 'Live patient' },
  { id: 'benchmarks', label: 'Benchmarks', icon: Activity, caption: 'Model metrics' },
  { id: 'audit', label: 'Audit', icon: ShieldCheck, caption: 'Paper vs ours' },
]

export default function App() {
  const [page, setPage] = useState<PageId>('diagnosis')
  const [health, setHealth] = useState<{ models: string[]; n_rows: number } | null>(null)
  const [healthErr, setHealthErr] = useState<string | null>(null)

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch((e) => setHealthErr(String(e)))
  }, [])

  return (
    <div className="flex min-h-screen">
      {/* ---------------------------------------------------------- sidebar */}
      <aside className="sticky top-0 flex h-screen w-60 shrink-0 flex-col border-r border-white/[0.06] bg-base-900/60 px-4 py-6 backdrop-blur-xl">
        <div className="mb-8 px-2">
          <h1 className="text-lg font-semibold tracking-tight text-white">
            Cardio<span className="text-gradient">Scan AI</span>
          </h1>
          <p className="mt-1 text-[11px] leading-relaxed text-slate-500">
            Cleveland heart-disease replication lab — 303 patients, six pipelines, zero leakage.
          </p>
        </div>

        <nav className="flex flex-col gap-1">
          {NAV.map(({ id, label, icon: Icon, caption }) => (
            <button
              key={id}
              onClick={() => setPage(id)}
              className={`group flex items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-all ${
                page === id
                  ? 'border border-accent-indigo/25 bg-accent-indigo/10 text-white'
                  : 'border border-transparent text-slate-400 hover:bg-white/[0.04] hover:text-slate-200'
              }`}
            >
              <Icon className={`h-4 w-4 ${page === id ? 'text-accent-indigo' : 'text-slate-500 group-hover:text-slate-300'}`} />
              <div>
                <p className="text-[13px] font-medium leading-tight">{label}</p>
                <p className="text-[10px] leading-tight text-slate-500">{caption}</p>
              </div>
            </button>
          ))}
        </nav>

        <div className="mt-auto px-2">
          {healthErr ? (
            <p className="text-[11px] leading-relaxed text-accent-rose">
              Backend offline — start it with<br />
              <code className="num text-[10px]">uvicorn backend.app:app</code>
            </p>
          ) : health ? (
            <div className="rounded-xl border border-white/[0.06] bg-white/[0.03] p-3">
              <p className="flex items-center gap-1.5 text-[11px] text-slate-400">
                <span className="inline-block h-1.5 w-1.5 rounded-full bg-accent-emerald shadow-[0_0_8px_rgba(52,211,153,0.8)]" />
                Backend live
              </p>
              <p className="num mt-1 text-[11px] text-slate-500">
                {health.models.length} models · {health.n_rows} patients
              </p>
            </div>
          ) : (
            <p className="text-[11px] text-slate-500">Connecting…</p>
          )}
          <p className="mt-3 text-[10px] text-slate-600">
            Models fit on the 70/30 train split only. Held-out metrics, never training scores.
          </p>
        </div>
      </aside>

      {/* ------------------------------------------------------------ main */}
      <main className="min-w-0 flex-1 px-8 py-8 lg:px-12">
        <div className="mx-auto max-w-6xl">
          {page === 'diagnosis' && <Diagnosis />}
          {page === 'benchmarks' && <Benchmarks />}
          {page === 'audit' && <Audit />}
        </div>
      </main>
    </div>
  )
}
