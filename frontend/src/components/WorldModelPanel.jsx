import { useCallback, useEffect, useRef, useState } from 'react';
import axios from 'axios';

const API_BASE = import.meta.env.VITE_WORLD_MODEL_URL || '';

function WhyList({ title, rows }) {
  if (!rows?.length) return null;
  return (
    <div className="flex-1 min-w-0">
      <div className="text-[9px] font-bold tracking-[0.2em] uppercase text-[#8B949E] mb-1.5">
        {title}
      </div>
      <ul className="space-y-1">
        {rows.slice(0, 5).map((row) => (
          <li
            key={`${title}-${row.name}`}
            className="flex items-center justify-between gap-2 font-mono text-[10px]"
          >
            <span className="truncate text-[#e0f2fe]">{row.name}</span>
            <span className={row.direction === 'down' ? 'text-[#FF8C00]' : 'text-[var(--color-accent)]'}>
              {row.direction === 'down' ? '▼' : '▲'} {Math.abs(Number(row.score)).toFixed(3)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Timeline({ steps }) {
  if (!steps?.length) return null;
  return (
    <div>
      <div className="text-[9px] font-bold tracking-[0.2em] uppercase text-[#8B949E] mb-1.5">
        30s timeline (5s steps)
      </div>
      <div className="space-y-1 font-mono text-[10px]">
        {steps.map((step) => (
          <div key={step.k} className="flex items-center justify-between gap-2 text-[#e0f2fe]">
            <span className="text-[#8B949E]">t+{step.seconds_ahead}s</span>
            <span className="truncate">
              {step.stage?.replaceAll('_', ' ')}
              {step.technique_id ? ` · ${step.technique_id}` : ''}
            </span>
            <span className="text-[var(--color-accent)]">P={Number(step.attack_probability).toFixed(3)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function WorldModelPanel() {
  const [bundle, setBundle] = useState(null);
  const [status, setStatus] = useState(null);
  const [index, setIndex] = useState(0);
  const [mode, setMode] = useState('replay');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const fileRef = useRef(null);

  const load = useCallback(async (i) => {
    setLoading(true);
    setError('');
    try {
      const [st, fc] = await Promise.all([
        axios.get(`${API_BASE}/api/world-model/status`),
        axios.get(`${API_BASE}/api/world-model/forecast`, { params: { i } }),
      ]);
      setStatus(st.data);
      setBundle(fc.data);
      setIndex(fc.data.index ?? i);
      setMode('replay');
    } catch {
      setError('World-model API offline. From Normnative-/ run: python -m src.world_model.api');
      setBundle(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load(0);
  }, [load]);

  const onUpload = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    setLoading(true);
    setError('');
    try {
      const form = new FormData();
      form.append('file', file);
      const res = await axios.post(`${API_BASE}/api/world-model/upload`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setBundle(res.data);
      setMode('upload');
    } catch (err) {
      const detail = err.response?.data?.detail;
      setError(typeof detail === 'string' ? detail : 'CSV upload failed. Use a CIC-IDS2018 flow CSV with Timestamp + Label.');
    } finally {
      setLoading(false);
    }
  };

  const n = bundle?.n_examples || status?.n_examples || 8;
  const prob = Number(bundle?.attack_probability ?? 0);
  const thr = Number(bundle?.attack_threshold ?? 0.15);
  const pct = Math.min(100, (prob / Math.max(thr, 1e-6)) * 100);
  const alert = Boolean(bundle?.something_bad);
  const isUpload = mode === 'upload' || bundle?.source === 'upload';

  return (
    <div className="panel border-[var(--color-accent)] shadow-[0_0_15px_rgba(0,212,255,0.08)]">
      <div className="panel-header flex items-center justify-between gap-2">
        <span>WORLD MODEL · 30s FORECAST</span>
        <span className="text-[9px] tracking-widest text-[#8B949E]">
          {status?.frozen ? 'LSTM FROZEN' : '—'} · {isUpload ? 'CSV UPLOAD' : '01 MAR REPLAY'}
        </span>
      </div>

      {error && (
        <div className="text-[11px] text-[var(--color-critical)] px-3 py-2 border-b border-[var(--color-border)]">
          {error}
        </div>
      )}

      {loading && !bundle && (
        <div className="px-3 py-4 text-[11px] text-[#8B949E] font-mono">Loading forecast…</div>
      )}

      {bundle && (
        <div className="p-3 space-y-3">
          <div className="flex items-start justify-between gap-3">
            <div>
              <div
                className={`text-[10px] font-black tracking-[0.18em] uppercase ${
                  alert ? 'text-[var(--color-critical)]' : 'text-[#00FF88]'
                }`}
              >
                {alert ? 'Something bad is likely' : 'No binary attack alert'}
              </div>
              <div className="mt-1 text-sm font-semibold text-white">
                What: {bundle.stage?.replaceAll('_', ' ')}
                {bundle.technique_id ? ` · ${bundle.technique_id}` : ''}
              </div>
              <div className="text-[10px] text-[#8B949E] font-mono">
                {bundle.technique_name || 'benign'}
                {isUpload
                  ? ` · ${bundle.source_file || 'upload'} · now ${bundle.t_now_iso || ''}`
                  : ` · window ${index + 1}/${n}`}
              </div>
            </div>
            <div className="flex gap-1 flex-none flex-wrap justify-end">
              <input ref={fileRef} type="file" accept=".csv,text/csv" className="hidden" onChange={onUpload} />
              <button
                type="button"
                className="px-2 py-1 text-[10px] font-bold border border-[var(--color-border)] rounded hover:bg-[rgba(0,229,255,0.08)]"
                onClick={() => fileRef.current?.click()}
              >
                UPLOAD CSV
              </button>
              {isUpload ? (
                <button
                  type="button"
                  className="px-2 py-1 text-[10px] font-bold border border-[var(--color-border)] rounded hover:bg-[rgba(0,229,255,0.08)]"
                  onClick={() => load(index)}
                >
                  REPLAY
                </button>
              ) : (
                <>
                  <button
                    type="button"
                    className="px-2 py-1 text-[10px] font-bold border border-[var(--color-border)] rounded hover:bg-[rgba(0,229,255,0.08)]"
                    onClick={() => load((index - 1 + n) % n)}
                  >
                    PREV
                  </button>
                  <button
                    type="button"
                    className="px-2 py-1 text-[10px] font-bold border border-[var(--color-border)] rounded hover:bg-[rgba(0,229,255,0.08)]"
                    onClick={() => load((index + 1) % n)}
                  >
                    NEXT
                  </button>
                </>
              )}
            </div>
          </div>

          <div>
            <div className="flex justify-between text-[9px] font-mono text-[#8B949E] mb-1">
              <span>P(attack in 30s) {prob.toFixed(3)}</span>
              <span>threshold {thr.toFixed(2)}</span>
            </div>
            <div className="h-2 rounded-full bg-[rgba(255,255,255,0.06)] overflow-hidden">
              <div
                className="h-full rounded-full"
                style={{
                  width: `${Math.min(pct, 100)}%`,
                  background: alert ? 'var(--color-critical)' : 'var(--color-accent)',
                }}
              />
            </div>
          </div>

          <p className="text-[11px] leading-relaxed text-[#e0f2fe] border border-[var(--color-border)] rounded p-2 bg-[rgba(0,229,255,0.03)]">
            {bundle.narrative}
          </p>

          <Timeline steps={bundle.timeline} />

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <WhyList title="Why · attack head" rows={bundle.why_attack} />
            <WhyList title="Why · MITRE stage" rows={bundle.why_stage} />
            <WhyList title="Why · 30s change" rows={bundle.why_change} />
          </div>
        </div>
      )}
    </div>
  );
}
