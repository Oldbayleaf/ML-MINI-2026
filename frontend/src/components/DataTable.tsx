import { toRows, type Table } from '../api'

/** Format a single cell for display: numbers get sensible precision, NaN → —. */
function fmt(v: string | number | null): string {
  if (v == null) return '—'
  if (typeof v === 'number') {
    if (Number.isNaN(v)) return '—'
    if (Number.isInteger(v)) return String(v)
    return Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(3)
  }
  const s = String(v)
  return s === 'nan' ? '—' : s
}

/** Highlight rule: good/bad tints for known metric columns. */
function tint(col: string, v: string | number | null): string {
  if (typeof v !== 'number' || Number.isNaN(v)) return ''
  const lower = col.toLowerCase()
  const goodHigh = lower.includes('accuracy') || lower.includes('auc') || lower.includes('f1') ||
    lower.includes('precision') || lower.includes('recall') || lower.includes('specificity') ||
    lower.includes('sensitivity') || lower.includes('npv') || lower.includes('ppv') || lower.includes('importance')
  const badHigh = lower.includes('error') || lower.includes('fp') || lower.includes('fn')
  if (goodHigh && v >= 0.8) return 'text-emerald-300'
  if (goodHigh && v >= 0.7) return 'text-slate-200'
  if (goodHigh) return 'text-slate-400'
  if (badHigh && v > 0) return 'text-rose-300/90'
  return ''
}

/**
 * Generic glass data table over a column-oriented backend Table.
 * Numeric cells are right-aligned and monospaced; the first column is treated as the key.
 */
export default function DataTable({ table, keyName, dense = false, maxRows }: {
  table: Table
  keyName?: string
  dense?: boolean
  maxRows?: number
}) {
  const rows = toRows(table)
  const shown = maxRows ? rows.slice(0, maxRows) : rows
  const first = table.columns[0]

  return (
    <div className="overflow-x-auto rounded-xl border border-white/[0.06]">
      <table className="w-full text-left text-xs">
        <thead>
          <tr className="border-b border-white/10 bg-white/[0.03]">
            {table.columns.map((c) => (
              <th
                key={c}
                className={`whitespace-nowrap px-3 py-2 font-mono text-[10px] uppercase tracking-wider text-slate-400 ${
                  c === first ? 'text-left' : 'text-right'
                }`}
              >
                {c.replace(/_/g, ' ')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {shown.map((row, i) => (
            <tr key={i} className="border-b border-white/[0.04] transition-colors last:border-0 hover:bg-white/[0.04]">
              {table.columns.map((c) => {
                const v = row[c]
                const isKey = c === (keyName ?? first)
                return (
                  <td
                    key={c}
                    className={`px-3 ${dense ? 'py-1.5' : 'py-2'} ${
                      isKey ? 'text-left font-medium text-slate-100' : `text-right text-slate-300 ${tint(c, v)}`
                    } ${typeof v === 'number' && !isKey ? 'num' : ''}`}
                  >
                    {isKey ? String(v ?? '—') : fmt(v)}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length === 0 && <p className="p-4 text-center text-xs text-slate-500">No rows.</p>}
    </div>
  )
}
