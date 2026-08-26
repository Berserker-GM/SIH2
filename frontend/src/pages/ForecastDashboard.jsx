import { useCallback, useEffect, useRef, useState } from 'react';
import axios from 'axios';

const API_BASE = import.meta.env.VITE_WORLD_MODEL_URL || '';
const UPLOAD_TIMEOUT_MS = 10 * 60 * 1000;

function fmtStage(stage) {
  return String(stage || '—').replaceAll('_', ' ');
}

function fmtBytes(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n < 0) return '';
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(0)} MB`;
  if (n >= 1024) return `${Math.max(1, Math.round(n / 1024))} KB`;
  return `${n} B`;
}

function fmtNum(value, digits = 3) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '—';
  if (Math.abs(n) >= 1000) return n.toExponential(2);
  return n.toFixed(digits);
}

function stageColor(stage) {
  const key = String(stage || 'benign');
  if (key === 'benign') return '#00FF88';
  if (key === 'reconnaissance') return '#00e5ff';
  if (key === 'initial_access') return '#ffcc00';
  if (key === 'lateral_movement') return '#ff6600';
  if (key === 'command_and_control') return '#ff003c';
  if (key === 'exfiltration') return '#c084fc';
  if (key === 'impact') return '#ff003c';
  return '#8B949E';
}

function vectorEnergy(row) {
  if (!Array.isArray(row) || row.length === 0) return 0;
  let acc = 0;
  for (const v of row) {
    const n = Number(v);
    if (Number.isFinite(n)) acc += n * n;
  }
  return Math.sqrt(acc);
}

function nodeFill(kind, isHub) {
  if (isHub) return '#ff6600';
  if (kind === 'host') return '#00e5ff';
  if (kind === 'service') return '#ffcc00';
  return '#8B949E';
}

function NetworkStateGraph({ graph }) {
  if (!graph || !Array.isArray(graph.nodes) || graph.nodes.length === 0) return null;
  const nodes = graph.nodes;
  const edges = Array.isArray(graph.edges) ? graph.edges : [];
  const width = 720;
  const height = 300;
  const cx = 360;
  const cy = 148;
  const hubId = graph.hub?.id;
  const center = nodes.reduce(
    (best, node) => (node.out_degree > (best?.out_degree ?? -1) ? node : best),
    nodes[0],
  );
  const others = nodes.filter((node) => node.id !== center.id);
  const pos = { [center.id]: { x: cx, y: cy } };
  others.forEach((node, i) => {
    const angle = ((2 * Math.PI * i) / Math.max(others.length, 1)) - Math.PI / 2;
    const radius = 108 + (i % 2) * 16;
    pos[node.id] = { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) };
  });
  const maxDeg = Math.max(1, ...nodes.map((node) => node.in_degree + node.out_degree));
  const modeLabel =
    graph.mode === 'host'
      ? 'host graph (Src IP → Dst IP)'
      : graph.mode === 'host_service'
        ? 'host → dest-port graph'
        : 'service graph (CIC ML: dest ports, sources unobserved)';

  return (
    <div className="panel overflow-hidden">
      <div className="panel-header">G_t · communication graph (same 5s window as S_t)</div>
      <div className="overflow-x-auto p-3">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full min-w-[720px] h-[300px]"
          role="img"
          aria-label="Per-window network communication graph"
        >
          {edges.map((edge, i) => {
            const a = pos[edge.src];
            const b = pos[edge.dst];
            if (!a || !b) return null;
            return (
              <line
                key={`ge${i}`}
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                stroke="#334155"
                strokeWidth={Math.max(1, Math.min(4, 0.6 + Number(edge.flows || 1) * 0.35))}
              />
            );
          })}
          {nodes.map((node) => {
            const p = pos[node.id];
            if (!p) return null;
            const r = 8 + (10 * (node.in_degree + node.out_degree)) / maxDeg;
            const isHub = node.id === hubId;
            return (
              <g key={node.id} transform={`translate(${p.x}, ${p.y})`}>
                <circle r={r} fill={nodeFill(node.kind, isHub)} fillOpacity="0.92" stroke="#e0f2fe" strokeWidth="1" />
                <text
                  y={r + 12}
                  textAnchor="middle"
                  fill="#e0f2fe"
                  fontSize="9"
                  fontFamily="ui-monospace, monospace"
                >
                  {String(node.label).slice(0, 18)}
                </text>
              </g>
            );
          })}
        </svg>
        <div className="mt-1 flex flex-wrap gap-3 font-mono text-[9px] text-[#8B949E]">
          <span>{modeLabel}</span>
          <span>nodes={graph.n_nodes} · edges={graph.n_edges} · flows={graph.n_flows}</span>
          <span>max out-degree (fan-out)={graph.max_out_degree}</span>
          <span>max in-degree (fan-in)={graph.max_in_degree}</span>
          {graph.hub ? <span>hub={graph.hub.label}</span> : null}
        </div>
        <p className="mt-2 text-[11px] leading-relaxed text-[#8B949E]">{graph.note}</p>
      </div>
    </div>
  );
}

function StateTransitionGraph({ history, timeline, nowLabel }) {
  const hist = Array.isArray(history) ? history : [];
  const future = Array.isArray(timeline) ? timeline : [];
  if (!hist.length && !future.length) return null;

  const observed = hist.map((row, i) => ({
    id: `h${i}`,
    kind: 'observed',
    label: i === hist.length - 1 ? 't' : `t-${hist.length - 1 - i}`,
    sub: i === hist.length - 1 ? (nowLabel || 'S_t') : 'history',
    energy: vectorEnergy(row),
    p: null,
    stage: i === hist.length - 1 ? 'now' : 'observed',
  }));
  const predicted = future.map((step) => ({
    id: `f${step.k}`,
    kind: 'forecast',
    label: `t+${step.seconds_ahead}s`,
    sub: fmtStage(step.stage),
    energy: null,
    p: Number(step.attack_probability),
    stage: step.stage,
  }));
  const nodes = [...observed, ...predicted];
  const n = nodes.length;
  const width = Math.max(720, n * 92);
  const height = 168;
  const pad = 48;
  const y = 78;
  const xs = nodes.map((_, i) => pad + (i * (width - pad * 2)) / Math.max(n - 1, 1));
  const maxE = Math.max(1, ...observed.map((node) => node.energy));

  return (
    <div className="panel overflow-hidden">
      <div className="panel-header">State graph · observed S → predicted Ŝ</div>
      <div className="overflow-x-auto p-3">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full min-w-[720px] h-[168px]"
          role="img"
          aria-label="Network state transition graph from 8 observed windows to 6 forecast steps"
        >
          {xs.slice(0, -1).map((x, i) => (
            <g key={`e${i}`}>
              <line
                x1={x + 18}
                y1={y}
                x2={xs[i + 1] - 18}
                y2={y}
                stroke={nodes[i + 1].kind === 'forecast' ? '#ff6600' : '#0088cc'}
                strokeWidth="2"
                strokeDasharray={nodes[i + 1].kind === 'forecast' ? '5 4' : '0'}
              />
            </g>
          ))}
          {nodes.map((node, i) => {
            const r = node.kind === 'observed'
              ? 10 + (8 * node.energy) / maxE
              : 11 + 10 * Math.min(1, Math.max(0, node.p || 0) / 0.15);
            const fill = node.kind === 'observed'
              ? (node.label === 't' ? '#00e5ff' : '#0ea5e9')
              : stageColor(node.stage);
            return (
              <g key={node.id} transform={`translate(${xs[i]}, ${y})`}>
                <circle r={r} fill={fill} fillOpacity="0.9" stroke="#e0f2fe" strokeWidth="1" />
                <text y={-28} textAnchor="middle" fill="#e0f2fe" fontSize="10" fontFamily="ui-monospace, monospace">
                  {node.label}
                </text>
                <text y={36} textAnchor="middle" fill="#8B949E" fontSize="9" fontFamily="ui-monospace, monospace">
                  {node.kind === 'forecast' ? `P=${(node.p || 0).toFixed(3)}` : node.sub}
                </text>
              </g>
            );
          })}
        </svg>
        <div className="mt-1 flex flex-wrap gap-3 font-mono text-[9px] text-[#8B949E]">
          <span>solid edges = observed 5s transitions</span>
          <span>dashed edges = LSTM closed-loop Ŝ</span>
          <span>observed node size = ||S|| (32-d energy)</span>
          <span>forecast node size / color = P(attack) and MITRE stage</span>
        </div>
      </div>
    </div>
  );
}

function WhyList({ title, rows }) {
  if (!rows?.length) return null;
  return (
    <div className="panel min-w-0">
      <div className="panel-header">{title}</div>
      <ul className="p-3 space-y-1.5">
        {rows.slice(0, 5).map((row) => (
          <li
            key={`${title}-${row.name}`}
            className="flex items-center justify-between gap-2 font-mono text-[11px]"
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

export default function ForecastDashboard() {
  const [status, setStatus] = useState(null);
  const [bundle, setBundle] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [pcapName, setPcapName] = useState('');
  const [localRel, setLocalRel] = useState('');
  const [sliceMode, setSliceMode] = useState('latest');
  const fileRef = useRef(null);
  const pcapRef = useRef(null);

  const pingStatus = useCallback(async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/world-model/status`, { timeout: 8000 });
      setStatus(res.data);
      return res.data;
    } catch {
      setStatus(null);
      return null;
    }
  }, []);

  useEffect(() => {
    pingStatus();
  }, [pingStatus]);

  const localFiles = Array.isArray(status?.local_csv) ? status.local_csv : [];
  const localDir = status?.local_csv_dir || 'data/raw/CIC-IDS2018';
  const localKey = localFiles.map((row) => row.rel).join('|');

  useEffect(() => {
    const files = Array.isArray(status?.local_csv) ? status.local_csv : [];
    if (!files.length) {
      setLocalRel('');
      return;
    }
    if (localRel && files.some((row) => row.rel === localRel)) return;
    const prefer =
      files.find((row) => /20-02-2018|loic/i.test(row.name || '')) ||
      files.find((row) => row.in_cic_folder) ||
      files[0];
    setLocalRel(prefer?.rel || '');
  }, [localKey, localRel, status]);

  const runLive = useCallback(async (request) => {
    setLoading(true);
    setError('');
    try {
      const st = await pingStatus();
      if (st && st.ready_for_live === false) {
        throw new Error(
          'Frozen LSTM is not loaded (missing world_lstm.pt / scaler.npz / train NPZ). Do not start a second API on :8001 — restart the existing one after the weights are present.',
        );
      }
      const res = await request();
      setBundle(res.data);
    } catch (err) {
      const detail = err.response?.data?.detail;
      setError(
        typeof detail === 'string'
          ? detail
          : err.message || 'Inference failed. Is python -m src.world_model.api running on :8001?',
      );
      setBundle(null);
    } finally {
      setLoading(false);
    }
  }, [pingStatus]);

  const onFile = useCallback(
    (file) => {
      if (!file) return;
      if (!String(file.name).toLowerCase().endsWith('.csv')) {
        setError('Upload a CIC-IDS2018 .csv file (Timestamp + Label).');
        return;
      }
      if (file.size > 80 * 1024 * 1024) {
        setError(
          'This CSV is larger than 80 MB. Keep large CIC days in data/raw/CIC-IDS2018 and pick them from the local-file dropdown.',
        );
        return;
      }
      const form = new FormData();
      form.append('file', file);
      const pcap = pcapRef.current?.files?.[0];
      if (pcap) {
        if (pcap.size > 80 * 1024 * 1024) {
          setError('Optional PCAP is larger than 80 MB.');
          return;
        }
        form.append('pcap', pcap);
      }
      runLive(() =>
        axios.post(`${API_BASE}/api/world-model/upload`, form, {
          timeout: UPLOAD_TIMEOUT_MS,
        }),
      );
    },
    [runLive],
  );

  const onInput = (event) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    onFile(file);
  };

  const onSample = () => {
    runLive(() =>
      axios.post(`${API_BASE}/api/world-model/sample`, null, { timeout: 120000 }),
    );
  };

  const onLocalRun = () => {
    if (!localRel) {
      setError(`No local CSV selected. Copy a TrafficForML file into ${localDir}, then refresh the list.`);
      return;
    }
    runLive(() =>
      axios.post(
        `${API_BASE}/api/world-model/local?name=${encodeURIComponent(localRel)}&slice_mode=${encodeURIComponent(sliceMode)}`,
        null,
        { timeout: UPLOAD_TIMEOUT_MS },
      ),
    );
  };

  const prob = Number(bundle?.attack_probability ?? 0);
  const thr = Number(bundle?.attack_threshold ?? status?.attack_threshold ?? 0.15);
  const pct = Math.min(100, (prob / Math.max(thr, 1e-6)) * 100);
  const alert = Boolean(bundle?.something_bad);
  const names = bundle?.feature_names || Object.keys(bundle?.s_t || {});
  const whyNames = new Set(
    [...(bundle?.why_attack || []), ...(bundle?.why_stage || []), ...(bundle?.why_change || [])]
      .map((row) => row.name),
  );
  const packetSpec = bundle?.packet_level_spec || status?.packet_level_spec || [];
  const packetUnit = Object.fromEntries(
    packetSpec.map((row) => [row.name, row.unit]),
  );
  const liveReady = status?.ready_for_live === true;

  return (
    <div className="flex-1 min-h-0 w-full overflow-y-auto custom-scrollbar">
      <header className="sticky top-0 z-10 border-b border-[var(--color-border)] bg-[rgba(2,8,19,0.92)] backdrop-blur px-5 py-3 flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="font-mono text-[11px] tracking-[0.28em] text-[var(--color-accent)]">
            NORMATIVE · WORLD MODEL
          </div>
          <h1 className="text-lg font-semibold text-white">CSV → 32-d vector + graph G_t → 30s forecast</h1>
        </div>
        <div className="flex flex-wrap items-center gap-2 font-mono text-[10px]">
          <span className="px-2 py-1 rounded border border-[var(--color-border)]">
            LSTM FROZEN
          </span>
          <span
            className={`px-2 py-1 rounded border ${
              liveReady
                ? 'border-[#00FF88] text-[#00FF88]'
                : 'border-[var(--color-critical)] text-[var(--color-critical)]'
            }`}
          >
            {status == null ? 'API OFFLINE' : liveReady ? 'LIVE INFERENCE READY' : 'LSTM NOT LOADED'}
          </span>
          <span className="px-2 py-1 rounded border border-[var(--color-border)] text-[#8B949E]">
            :8001 · K=6 · thr {thr.toFixed(2)}
          </span>
        </div>
      </header>

      <main className="max-w-6xl mx-auto p-5 space-y-4">
        <section className="panel p-4">
          <p className="text-[13px] leading-relaxed text-[#e0f2fe]">
            Upload a CICFlowMeter CSV. The API bins it into 5-second states: a frozen 32-d feature
            vector S_t plus a communication graph G_t for the same window. The latest 8 vectors
            (40s) roll the frozen LSTM six steps ahead. Output is attack probability, MITRE stage,
            and why — not the saved 01 Mar replay JSON.
          </p>
          <input
            ref={fileRef}
            type="file"
            accept=".csv,text/csv"
            className="hidden"
            onChange={onInput}
          />
          <div
            className={`mt-4 rounded-lg border-2 border-dashed px-4 py-8 text-center transition-colors ${
              dragOver
                ? 'border-[var(--color-accent)] bg-[rgba(0,229,255,0.08)]'
                : 'border-[var(--color-border)] bg-[rgba(0,229,255,0.02)]'
            }`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              onFile(e.dataTransfer.files?.[0]);
            }}
          >
            <div className="text-sm font-semibold text-white">Drop CIC-IDS2018 CSV here</div>
            <div className="mt-1 text-[11px] text-[#8B949E] font-mono">
              Needs Timestamp + Label. Optional PCAP fills TTL / fragment / retransmit
              as a sidecar (not injected into the frozen 32-d LSTM). Files over 80 MB
              cannot be uploaded in the browser — copy them into {localDir} and pick
              from the dropdown.
            </div>
            <div className="mt-3 font-mono text-[10px] text-[#8B949E]">
              Optional PCAP: {pcapName || 'none'}
              <button
                type="button"
                className="ml-2 underline"
                onClick={() => pcapRef.current?.click()}
              >
                attach .pcap
              </button>
            </div>
            <input
              ref={pcapRef}
              type="file"
              accept=".pcap,.pcapng,.cap"
              className="hidden"
              onChange={(event) => {
                setPcapName(event.target.files?.[0]?.name || '');
              }}
            />
            <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
              <button
                type="button"
                disabled={loading}
                className="px-3 py-1.5 text-[11px] font-bold rounded border border-[var(--color-accent)] text-[var(--color-accent)] hover:bg-[rgba(0,229,255,0.1)] disabled:opacity-50"
                onClick={() => fileRef.current?.click()}
              >
                {loading ? 'RUNNING FROZEN LSTM…' : 'UPLOAD CSV'}
              </button>
              <button
                type="button"
                disabled={loading}
                className="px-3 py-1.5 text-[11px] font-bold rounded border border-[var(--color-border)] hover:bg-[rgba(255,255,255,0.05)] disabled:opacity-50"
                onClick={onSample}
              >
                RUN SYNTHETIC SAMPLE
              </button>
            </div>
            <div className="mt-3 flex flex-wrap items-center justify-center gap-2">
              <select
                aria-label="Local CIC CSV on disk"
                value={localRel}
                disabled={loading || localFiles.length === 0}
                onFocus={() => pingStatus()}
                onChange={(event) => setLocalRel(event.target.value)}
                className="max-w-[min(100%,32rem)] min-w-[16rem] bg-[#020813] border border-[#ff6600] text-[#ffcc99] text-[11px] font-mono rounded px-2 py-1.5 disabled:opacity-50"
              >
                {localFiles.length === 0 ? (
                  <option value="">No CSVs in data/raw — copy one into CIC-IDS2018</option>
                ) : (
                  localFiles.map((row) => (
                    <option key={row.rel} value={row.rel}>
                      {row.rel} ({fmtBytes(row.bytes)})
                    </option>
                  ))
                )}
              </select>
              <select
                aria-label="How to pick the 90s live slice"
                value={sliceMode}
                disabled={loading}
                onChange={(event) => setSliceMode(event.target.value)}
                className="bg-[#020813] border border-[var(--color-border)] text-[#e0f2fe] text-[11px] font-mono rounded px-2 py-1.5 disabled:opacity-50"
              >
                <option value="latest">slice: latest 90s (now)</option>
                <option value="attack">slice: densest attack 90s</option>
              </select>
              <button
                type="button"
                disabled={loading || !localRel}
                className="px-3 py-1.5 text-[11px] font-bold rounded border border-[#ff6600] text-[#ff6600] hover:bg-[rgba(255,102,0,0.1)] disabled:opacity-50"
                onClick={onLocalRun}
              >
                RUN LOCAL CSV
              </button>
              <button
                type="button"
                disabled={loading}
                className="px-2 py-1.5 text-[10px] font-mono underline text-[#8B949E] disabled:opacity-50"
                onClick={() => pingStatus()}
              >
                refresh list
              </button>
            </div>
          </div>
        </section>

        {error && (
          <div className="panel border-[var(--color-critical)] px-4 py-3 text-[12px] text-[var(--color-critical)]">
            {error}
          </div>
        )}

        {loading && (
          <div className="panel px-4 py-3 font-mono text-[12px] text-[#8B949E]">
            First local run on a multi-GB unsorted CIC day scans the file and caches a
            90s slice (can take a few minutes). After that: 32-d windows → last 8 →
            closed-loop K=6…
          </div>
        )}

        {bundle && (
          <>
            <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="panel p-4">
                <div className="text-[10px] tracking-[0.2em] uppercase text-[#8B949E]">
                  Attack in 30s
                </div>
                <div
                  className={`mt-1 text-3xl font-black ${
                    alert ? 'text-[var(--color-critical)]' : 'text-[#00FF88]'
                  }`}
                >
                  {(prob * 100).toFixed(1)}%
                </div>
                <div className="mt-2 h-2 rounded-full bg-[rgba(255,255,255,0.06)] overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${Math.min(pct, 100)}%`,
                      background: alert ? 'var(--color-critical)' : 'var(--color-accent)',
                    }}
                  />
                </div>
                <div className="mt-2 font-mono text-[10px] text-[#8B949E]">
                  P={prob.toFixed(3)} · threshold {thr.toFixed(2)} ·{' '}
                  {alert ? 'ALERT' : 'below threshold'}
                  {bundle.s_t_attack_now != null ? (
                    <>
                      {' '}
                      · CIC bin {Number(bundle.s_t_attack_now) ? 'ATTACK' : 'Benign'}
                      {bundle.s_t_cic_stage ? ` (${fmtStage(bundle.s_t_cic_stage)})` : ''}
                    </>
                  ) : null}
                </div>
              </div>
              <div className="panel p-4">
                <div className="text-[10px] tracking-[0.2em] uppercase text-[#8B949E]">
                  MITRE stage
                </div>
                <div className="mt-1 text-xl font-semibold text-white capitalize">
                  {fmtStage(bundle.stage)}
                </div>
                <div className="mt-1 font-mono text-[12px] text-[var(--color-accent)]">
                  {bundle.technique_id || '—'} {bundle.technique_name || ''}
                </div>
                <div className="mt-2 font-mono text-[10px] text-[#8B949E]">
                  {bundle.source_file || bundle.source} · {bundle.n_windows ?? '—'} windows · last 8 fed to LSTM
                </div>
              </div>
              <div className="panel p-4">
                <div className="text-[10px] tracking-[0.2em] uppercase text-[#8B949E]">
                  Now (S_t)
                </div>
                <div className="mt-1 font-mono text-[12px] text-white">
                  {bundle.t_now_iso || '—'}
                </div>
                <div className="mt-2 font-mono text-[10px] text-[#8B949E]">
                  source={bundle.source} · seq {bundle.seq_len || 8}×{bundle.input_dim || 32} · G_t {bundle.state_graph?.mode || '—'} · horizon {bundle.horizon_seconds || 30}s
                </div>
                {bundle.live?.note ? (
                  <div className="mt-2 font-mono text-[10px] text-[#ffcc00]">
                    {bundle.live.truncated ? 'sliced · ' : ''}{bundle.live.note}
                    {bundle.live.n_rows != null ? ` · ${bundle.live.n_rows} flows` : ''}
                  </div>
                ) : null}
              </div>
            </section>

            <section className="panel p-4">
              <div className="text-[10px] tracking-[0.2em] uppercase text-[#8B949E] mb-2">Why</div>
              <p className="text-[13px] leading-relaxed text-[#e0f2fe]">{bundle.narrative}</p>
            </section>

            <NetworkStateGraph graph={bundle.state_graph} />

            <StateTransitionGraph
              history={bundle.history_states}
              timeline={bundle.timeline}
              nowLabel={bundle.t_now_iso}
            />

            <section className="panel overflow-hidden">
              <div className="panel-header">30s timeline (closed-loop Ŝ)</div>
              <div className="p-3 space-y-1 font-mono text-[11px]">
                {(bundle.timeline || []).map((step) => (
                  <div key={step.k} className="flex items-center justify-between gap-2 text-[#e0f2fe]">
                    <span className="text-[#8B949E]">t+{step.seconds_ahead}s</span>
                    <span className="truncate capitalize">
                      {fmtStage(step.stage)}
                      {step.technique_id ? ` · ${step.technique_id}` : ''}
                    </span>
                    <span className="text-[var(--color-accent)]">
                      P={Number(step.attack_probability).toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            </section>

            <section className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <WhyList title="Why · attack head" rows={bundle.why_attack} />
              <WhyList title="Why · MITRE stage" rows={bundle.why_stage} />
              <WhyList title="Why · 30s change" rows={bundle.why_change} />
            </section>

            <section className="panel overflow-hidden">
              <div className="panel-header">S_t · 32-d state (last 5s window, unscaled) · packet-level = indices 22–31</div>
              <div className="p-3 grid grid-cols-2 sm:grid-cols-4 gap-2">
                {names.map((name, i) => (
                  <div
                    key={name}
                    className={`rounded border px-2 py-1.5 ${
                      whyNames.has(name)
                        ? 'border-[var(--color-accent)] bg-[rgba(0,229,255,0.08)]'
                        : 'border-[var(--color-border)]'
                    }`}
                  >
                    <div className="font-mono text-[9px] text-[#8B949E] truncate">
                      {i} · {name}
                    </div>
                    <div className="font-mono text-[12px] text-white">
                      {fmtNum(bundle.s_t?.[name])}
                    </div>
                    {packetUnit[name] ? (
                      <div className="font-mono text-[8px] text-[#8B949E] truncate">{packetUnit[name]}</div>
                    ) : null}
                  </div>
                ))}
              </div>
            </section>

            {bundle.pcap_stats && (
              <section className="panel p-4">
                <div className="panel-header !px-0 !border-0">PCAP sidecar · dims 29–31 (not fed to LSTM)</div>
                <div className="mt-2 grid grid-cols-1 sm:grid-cols-3 gap-2 font-mono text-[11px]">
                  <div>ttl_variance = {fmtNum(bundle.pcap_stats.ttl_variance)} TTL²</div>
                  <div>ip_fragment_flags = {fmtNum(bundle.pcap_stats.ip_fragment_flags, 0)} packets</div>
                  <div>retransmit_count = {fmtNum(bundle.pcap_stats.retransmit_count, 0)} packets</div>
                </div>
                <p className="mt-2 text-[11px] text-[#8B949E]">{bundle.pcap_stats.note}</p>
              </section>
            )}

            {Array.isArray(bundle.history_states) && bundle.history_states.length > 0 && (
              <section className="panel overflow-hidden">
                <div className="panel-header">History fed to LSTM (8 × 32, unscaled)</div>
                <div className="overflow-x-auto p-3">
                  <table className="min-w-full font-mono text-[9px] text-[#e0f2fe]">
                    <thead>
                      <tr>
                        <th className="text-left pr-2 text-[#8B949E]">t</th>
                        {names.map((name) => (
                          <th key={name} className="text-right px-1 text-[#8B949E] font-normal">
                            {name}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {bundle.history_states.map((row, i) => (
                        <tr key={i} className={i === bundle.history_states.length - 1 ? 'text-[var(--color-accent)]' : ''}>
                          <td className="pr-2 whitespace-nowrap text-[#8B949E]">
                            t-{bundle.history_states.length - 1 - i}
                          </td>
                          {row.map((v, j) => (
                            <td key={j} className="text-right px-1 whitespace-nowrap">
                              {fmtNum(v, 2)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            )}
          </>
        )}
      </main>
    </div>
  );
}
