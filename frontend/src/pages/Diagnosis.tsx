import { useEffect, useState } from 'react'
import { HeartPulse, RotateCcw, Sparkles } from 'lucide-react'
import { api, pct, type ModelMeta, type Patient, type PredictResponse } from '../api'
import { Badge, Card, ErrorBox, Meter, PageHeader, Spinner } from '../components/ui'

/** Typical patient preset (58y male, asymptomatic, resting ECG normal-ish). */
const PRESET: Patient = {
  age: 58, sex: 1, cp: 4, trestbps: 138, chol: 250, fbs: 1, restecg: 0,
  thalach: 130, exang: 0, oldpeak: 1.2, slope: 2, ca: 0, thal: 3,
}

interface FieldSpec {
  key: keyof Patient
  label: string
  unit: string
  min: number
  max: number
  step: number
  help?: string
}

const NUMERIC_FIELDS: FieldSpec[] = [
  { key: 'age', label: 'Age', unit: 'years', min: 29, max: 77, step: 1 },
  { key: 'trestbps', label: 'Resting BP', unit: 'mm Hg', min: 94, max: 200, step: 1 },
  { key: 'chol', label: 'Cholesterol', unit: 'mg/dl', min: 126, max: 564, step: 1 },
  { key: 'thalach', label: 'Max heart rate', unit: 'bpm', min: 71, max: 202, step: 1 },
  { key: 'oldpeak', label: 'ST depression', unit: 'oldpeak', min: 0, max: 6.2, step: 0.1 },
]

interface ChoiceSpec {
  key: keyof Patient
  label: string
  options: { value: number; label: string }[]
  help?: string
}

const CHOICE_FIELDS: ChoiceSpec[] = [
  { key: 'sex', label: 'Sex', options: [{ value: 0, label: 'Female' }, { value: 1, label: 'Male' }] },
  {
    key: 'cp', label: 'Chest pain type',
    help: 'cp 1–4: typical angina, atypical, non-anginal, asymptomatic',
    options: [
      { value: 1, label: '1 · Typical angina' }, { value: 2, label: '2 · Atypical' },
      { value: 3, label: '3 · Non-anginal' }, { value: 4, label: '4 · Asymptomatic' },
    ],
  },
  { key: 'fbs', label: 'Fasting glucose > 120', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
  {
    key: 'restecg', label: 'Resting ECG',
    options: [{ value: 0, label: '0 · Normal' }, { value: 1, label: '1 · ST-T abnormality' }, { value: 2, label: '2 · LVH' }],
  },
  { key: 'exang', label: 'Exercise-induced angina', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
  {
    key: 'slope', label: 'ST slope',
    options: [{ value: 1, label: '1 · Upsloping' }, { value: 2, label: '2 · Flat' }, { value: 3, label: '3 · Downsloping' }],
  },
  { key: 'ca', label: 'Major vessels (0–3)', options: [0, 1, 2, 3].map((v) => ({ value: v, label: String(v) })) },
  {
    key: 'thal', label: 'Thalassemia',
    help: '3 = normal, 6 = fixed defect, 7 = reversible defect',
    options: [{ value: 3, label: '3 · Normal' }, { value: 6, label: '6 · Fixed defect' }, { value: 7, label: '7 · Reversible' }],
  },
]

const toneFor = (p: number) => (p < 0.4 ? 'good' : p < 0.6 ? 'warn' : 'bad') as 'good' | 'warn' | 'bad'

export default function Diagnosis() {
  const [patient, setPatient] = useState<Patient>(PRESET)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [meta, setMeta] = useState<ModelMeta[]>([])
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    api.modelsMeta().then((m) => setMeta(m.models)).catch((e) => setError(String(e)))
  }, [])

  const predict = () => {
    setBusy(true)
    setError(null)
    api
      .predict(patient)
      .then(setResult)
      .catch((e) => setError(String(e)))
      .finally(() => setBusy(false))
  }

  // auto-predict on mount so the page is never empty
  useEffect(() => {
    predict()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const set = (key: keyof Patient, value: number) => setPatient((p) => ({ ...p, [key]: value }))
  const consensusTone = result ? toneFor(result.consensus.mean_probability) : 'good'

  return (
    <div>
      <PageHeader
        eyebrow="Live inference"
        title="Patient Diagnosis"
        description="Adjust the clinical measurements and watch all six pipelines vote — calibrated probabilities, honest held-out scores."
      />

      <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
        {/* ------------------------------------------------------- the form */}
        <Card
          title="Patient measurements"
          subtitle="Range-checked against the Cleveland cohort"
          right={
            <button
              onClick={() => { setPatient(PRESET); predict() }}
              className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/[0.03] px-2.5 py-1.5 text-xs text-slate-300 transition-colors hover:bg-white/[0.07]"
            >
              <RotateCcw className="h-3 w-3" /> Reset preset
            </button>
          }
        >
          <div className="space-y-4">
            {NUMERIC_FIELDS.map((f) => (
              <div key={f.key}>
                <div className="mb-1 flex items-baseline justify-between">
                  <label className="text-xs font-medium text-slate-300">{f.label}</label>
                  <span className="num text-sm font-semibold text-white">
                    {patient[f.key]} <span className="text-[10px] font-normal text-slate-500">{f.unit}</span>
                  </span>
                </div>
                <input
                  type="range"
                  min={f.min}
                  max={f.max}
                  step={f.step}
                  value={patient[f.key]}
                  onChange={(e) => set(f.key, Number(e.target.value))}
                  className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-accent-indigo"
                />
              </div>
            ))}

            <div className="grid grid-cols-2 gap-4 pt-1">
              {CHOICE_FIELDS.map((f) => (
                <div key={f.key}>
                  <label className="mb-1 block text-xs font-medium text-slate-300">{f.label}</label>
                  <select
                    value={patient[f.key]}
                    onChange={(e) => set(f.key, Number(e.target.value))}
                    className="w-full rounded-lg border border-white/10 bg-base-850 px-2.5 py-2 text-xs text-slate-200 outline-none transition-colors focus:border-accent-indigo/40"
                  >
                    {f.options.map((o) => (
                      <option key={o.value} value={o.value} className="bg-base-850">{o.label}</option>
                    ))}
                  </select>
                  {f.help && <p className="mt-1 text-[10px] text-slate-500">{f.help}</p>}
                </div>
              ))}
            </div>

            <button
              onClick={predict}
              disabled={busy}
              className="mt-2 flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-accent-indigo/90 to-accent-cyan/70 px-4 py-2.5 text-sm font-semibold text-white transition-all hover:brightness-110 disabled:opacity-60"
            >
              <Sparkles className="h-4 w-4" />
              {busy ? 'Predicting…' : 'Run diagnosis'}
            </button>
          </div>
        </Card>

        {/* ------------------------------------------------------ the result */}
        <div className="space-y-6">
          {error && <ErrorBox error={error} />}
          {!result && !error && <Spinner label="Training models and calibrating…" />}

          {result && (
            <>
              <Card className="animate-fade-in">
                <div className="flex items-center gap-5">
                  <div
                    className={`flex h-20 w-20 shrink-0 items-center justify-center rounded-2xl border ${
                      consensusTone === 'bad'
                        ? 'border-accent-rose/40 bg-accent-rose/10'
                        : consensusTone === 'warn'
                          ? 'border-accent-amber/40 bg-accent-amber/10'
                          : 'border-accent-emerald/40 bg-accent-emerald/10'
                    }`}
                  >
                    <HeartPulse
                      className={`h-9 w-9 ${
                        consensusTone === 'bad' ? 'text-accent-rose' : consensusTone === 'warn' ? 'text-accent-amber' : 'text-accent-emerald'
                      }`}
                    />
                  </div>
                  <div className="min-w-0">
                    <p className="label-mono">Panel consensus</p>
                    <p className="mt-1 text-2xl font-semibold capitalize text-white">
                      {result.consensus.label === 'disease' ? 'Disease likely' : result.consensus.label === 'healthy' ? 'Likely healthy' : 'Borderline'}
                      {result.consensus.label === 'borderline' && <Badge tone="warn"><span className="ml-1">needs review</span></Badge>}
                    </p>
                    <p className="mt-1 text-xs text-slate-400">
                      <span className="num">{result.consensus.votes_for_disease}</span> of{' '}
                      <span className="num">{result.consensus.total}</span> models vote disease · mean calibrated P(disease){' '}
                      <span className="num">{pct(result.consensus.mean_probability)}</span>
                    </p>
                  </div>
                </div>
              </Card>

              <Card title="Per-model verdicts" subtitle="Hard vote + calibrated probability">
                <div className="space-y-3.5">
                  {result.models.map((m) => {
                    const t = toneFor(m.probability)
                    return (
                      <div key={m.model} className="grid grid-cols-[130px_1fr_auto] items-center gap-3">
                        <div>
                          <p className="text-[13px] font-medium text-slate-100">{m.model}</p>
                          <p className="num text-[10px] text-slate-500">
                            held-out {pct(m.held_out_accuracy, 1)} ± {pct(m.ci_half, 1)}
                          </p>
                        </div>
                        <Meter value={m.probability} tone={t} />
                        <div className="text-right">
                          <p className={`num text-sm font-semibold ${t === 'bad' ? 'text-rose-300' : t === 'warn' ? 'text-amber-300' : 'text-emerald-300'}`}>
                            {pct(m.probability, 1)}
                          </p>
                          <p className="text-[10px] text-slate-500">{m.prediction === 1 ? 'disease' : 'healthy'}</p>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </Card>
            </>
          )}

          {meta.length > 0 && (
            <p className="text-[11px] leading-relaxed text-slate-500">
              Held-out accuracies from the untouched test split (n={meta[0].n_test}): these are the
              honest numbers each model earns on unseen patients — not its training score. On 303
              patients the 95% CI is roughly ±{pct(meta[0].ci_half, 0)}, so treat any single prediction as a
              screening signal, not a diagnosis.
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
