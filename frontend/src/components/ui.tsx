import type { ReactNode } from 'react'
import { AlertTriangle, Loader2 } from 'lucide-react'

/* ------------------------------------------------------------------ layout */
export function Card({ title, subtitle, right, children, className = '' }: {
  title?: ReactNode
  subtitle?: ReactNode
  right?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section className={`glass p-5 ${className}`}>
      {(title || right) && (
        <header className="mb-4 flex items-start justify-between gap-4">
          <div>
            {title && <h3 className="text-sm font-semibold text-slate-100">{title}</h3>}
            {subtitle && <p className="mt-0.5 text-xs text-slate-400">{subtitle}</p>}
          </div>
          {right}
        </header>
      )}
      {children}
    </section>
  )
}

export function PageHeader({ eyebrow, title, description }: {
  eyebrow: string
  title: string
  description: string
}) {
  return (
    <header className="mb-6 animate-fade-in">
      <p className="label-mono">{eyebrow}</p>
      <h1 className="mt-1.5 text-2xl font-semibold tracking-tight text-white">{title}</h1>
      <p className="mt-1.5 max-w-3xl text-sm text-slate-400">{description}</p>
    </header>
  )
}

/* ------------------------------------------------------------------ states */
export function Spinner({ label = 'Crunching models…' }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 py-10 text-sm text-slate-400">
      <Loader2 className="h-4 w-4 animate-spin text-accent-indigo" />
      {label}
    </div>
  )
}

export function ErrorBox({ error }: { error: string }) {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-accent-rose/25 bg-accent-rose/[0.07] p-4 text-sm text-rose-200">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-accent-rose" />
      <div>
        <p className="font-medium">Something went wrong</p>
        <p className="mt-0.5 text-rose-200/70">{error}</p>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ pieces */
export function Badge({ children, tone = 'slate' }: {
  children: ReactNode
  tone?: 'slate' | 'good' | 'warn' | 'bad' | 'indigo' | 'cyan'
}) {
  const tones = {
    slate: 'border-white/10 bg-white/5 text-slate-300',
    good: 'border-accent-emerald/30 bg-accent-emerald/10 text-emerald-200',
    warn: 'border-accent-amber/30 bg-accent-amber/10 text-amber-200',
    bad: 'border-accent-rose/30 bg-accent-rose/10 text-rose-200',
    indigo: 'border-accent-indigo/30 bg-accent-indigo/10 text-indigo-200',
    cyan: 'border-accent-cyan/30 bg-accent-cyan/10 text-cyan-200',
  } as const
  return <span className={`chip ${tones[tone]}`}>{children}</span>
}

/** Segmented tab switcher used across Benchmarks / Audit / Lab. */
export function Tabs<T extends string>({ options, value, onChange }: {
  options: { id: T; label: string }[]
  value: T
  onChange: (id: T) => void
}) {
  return (
    <div className="inline-flex flex-wrap gap-1 rounded-xl border border-white/10 bg-white/[0.03] p-1">
      {options.map((o) => (
        <button
          key={o.id}
          onClick={() => onChange(o.id)}
          className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
            value === o.id
              ? 'bg-accent-indigo/20 text-indigo-200 shadow-[inset_0_0_0_1px_rgba(129,140,248,0.3)]'
              : 'text-slate-400 hover:bg-white/5 hover:text-slate-200'
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

/** Horizontal probability bar used on the Diagnosis page. */
export function Meter({ value, tone }: { value: number; tone: 'good' | 'warn' | 'bad' }) {
  const colors = {
    good: 'from-accent-emerald/80 to-accent-emerald/40',
    warn: 'from-accent-amber/80 to-accent-amber/40',
    bad: 'from-accent-rose/80 to-accent-rose/40',
  } as const
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/[0.06]">
      <div
        className={`h-full rounded-full bg-gradient-to-r ${colors[tone]} transition-[width] duration-500`}
        style={{ width: `${Math.max(1, Math.min(100, value * 100))}%` }}
      />
    </div>
  )
}

