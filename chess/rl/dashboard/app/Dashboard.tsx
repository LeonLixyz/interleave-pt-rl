"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

type Point = {
  step: number;
  reward?: number;
  pass1?: number;
  pass2?: number;
  pass4?: number;
  pass8?: number;
  entropy?: number;
  gradNorm?: number;
  ppoKl?: number;
  allZeroPercentage?: number;
};

type Run = {
  id: string;
  label: string;
  shortLabel: string;
  legacyRunId: string;
  appId: string;
  callId: string;
  color: string;
  status: "running" | "complete" | "stopped";
  latestStep: number;
  checkpoint: number;
  points: Point[];
};

type DashboardData = {
  generatedAt: string;
  targetStep: number;
  refreshSeconds: number;
  dataLabel: string;
  modelLabel: string;
  comparisonNote: string;
  source: string;
  runs: Run[];
};

type Metric = "reward" | "pass1" | "pass2" | "pass4" | "pass8" | "entropy" | "gradNorm" | "ppoKl" | "allZeroPercentage";
type MetricScale = "unit" | "nonnegative" | "signed";
type ChartPoint = { step: number; value: number };

const metricConfig: Record<Metric, { label: string; scale: MetricScale; note: string }> = {
  reward: { label: "Raw reward", scale: "unit", note: "Mean binary reward over the sampled trajectories in each RL update." },
  pass1: { label: "Pass@1", scale: "unit", note: "Estimated probability that one sample succeeds." },
  pass2: { label: "Pass@2", scale: "unit", note: "Estimated probability that at least one of two samples succeeds." },
  pass4: { label: "Pass@4", scale: "unit", note: "Estimated probability that at least one of four samples succeeds." },
  pass8: { label: "Pass@8", scale: "unit", note: "Estimated probability that at least one of eight samples succeeds." },
  entropy: { label: "Policy entropy", scale: "nonnegative", note: "Mean rollout policy entropy logged for each update." },
  gradNorm: { label: "Gradient norm", scale: "nonnegative", note: "The logged train/grad_norm value for each optimizer update." },
  ppoKl: { label: "PPO KL", scale: "signed", note: "The logged train/ppo_kl statistic, not the KL penalty loss. It can remain at or near zero for this on-policy update geometry." },
  allZeroPercentage: { label: "All-zero groups", scale: "unit", note: "Fraction of prompt groups whose eight sampled rewards are all zero." },
};

const fallbackData: DashboardData = {
  generatedAt: new Date(0).toISOString(),
  targetStep: 1500,
  refreshSeconds: 30,
  dataLabel: "mixed-outcome filtered (1–15/16)",
  modelLabel: "Qwen3 · 47,245,312 parameters",
  comparisonNote: "These checkpoints are not exact PT/SFT recipe matches. Pretraining differs in context handling, vocabulary, batch size, learning-rate schedule, and data selection; SFT differs in effective batch size, precision, weight decay, and exact rows. Within each checkpoint, the two RL learning-rate arms are matched.",
  source: "Modal application logs",
  runs: [],
};

const metricsUrl = "https://modal-labs-leon-dev--chess-rl-pretrain-sft-rl-comparison-fda3ac.modal.run";

function finiteMetric(point: Point, metric: Metric) {
  const value = point[metric];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function movingAverage(points: Point[], metric: Metric, window: number): ChartPoint[] {
  const source = points.flatMap((point) => {
    const value = finiteMetric(point, metric);
    return value === undefined ? [] : [{ step: point.step, value }];
  });
  if (window === 1) return source;
  let sum = 0;
  return source.map((point, index) => {
    sum += point.value;
    if (index >= window) sum -= source[index - window].value;
    return { step: point.step, value: sum / Math.min(index + 1, window) };
  });
}

function nearestPoint(points: ChartPoint[], step: number) {
  if (!points.length) return undefined;
  let low = 0;
  let high = points.length - 1;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (points[middle].step < step) low = middle + 1;
    else high = middle;
  }
  const candidate = points[low];
  const previous = points[Math.max(0, low - 1)];
  return Math.abs(previous.step - step) < Math.abs(candidate.step - step) ? previous : candidate;
}

function metricDomain(values: number[], scale: MetricScale): [number, number] {
  if (scale === "unit") return [0, 1];
  if (!values.length) return [0, 1];
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  if (scale === "nonnegative") return [0, maximum > 0 ? maximum * 1.05 : 1];
  if (minimum === 0 && maximum === 0) return [-1e-6, 1e-6];
  if (minimum >= 0) return [0, maximum * 1.05];
  if (maximum <= 0) return [minimum * 1.05, 0];
  const bound = Math.max(Math.abs(minimum), Math.abs(maximum)) * 1.05;
  return [-bound, bound];
}

function formatMetric(value: number, metric: Metric, axis = false) {
  if (metric === "allZeroPercentage") return `${(value * 100).toFixed(axis ? 0 : 1)}%`;
  if (metric === "ppoKl") {
    if (value === 0) return "0";
    return Math.abs(value) < 0.001 ? value.toExponential(axis ? 1 : 3) : value.toFixed(axis ? 3 : 6);
  }
  if (metric === "reward" || metric.startsWith("pass")) return value.toFixed(axis ? 1 : 3);
  if (axis) {
    if (Math.abs(value) >= 10) return value.toFixed(0);
    if (Math.abs(value) >= 1) return value.toFixed(1);
    if (Math.abs(value) >= 0.01) return value.toFixed(2);
    return value === 0 ? "0" : value.toExponential(1);
  }
  return value.toPrecision(5);
}

function latestPointWithMetric(points: Point[], metric: Metric) {
  return points.findLast((point) => finiteMetric(point, metric) !== undefined);
}

function CurveChart({
  runs,
  targetStep,
  metric,
  smoothing,
  visible,
}: {
  runs: Run[];
  targetStep: number;
  metric: Metric;
  smoothing: number;
  visible: Set<string>;
}) {
  const width = 1000;
  const height = 430;
  const margin = { top: 22, right: 24, bottom: 48, left: 58 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;
  const [hoverStep, setHoverStep] = useState<number | null>(null);

  const x = (step: number) => margin.left + (step / targetStep) * chartWidth;
  const visibleRuns = runs.filter((run) => visible.has(run.id));
  const chartRuns = visibleRuns.map((run) => ({
    run,
    raw: movingAverage(run.points, metric, 1),
    smoothed: movingAverage(run.points, metric, smoothing),
  }));
  const [domainMinimum, domainMaximum] = metricDomain(
    chartRuns.flatMap(({ raw }) => raw.map((point) => point.value)),
    metricConfig[metric].scale,
  );
  const y = (value: number) => margin.top + ((domainMaximum - value) / (domainMaximum - domainMinimum)) * chartHeight;
  const ticksX = [0, 300, 600, 900, 1200, 1500];
  const ticksY = Array.from({ length: 6 }, (_, index) => domainMinimum + ((domainMaximum - domainMinimum) * index) / 5);

  const tooltip = hoverStep === null
    ? []
    : chartRuns.map(({ run, raw }) => ({ run, point: nearestPoint(raw, hoverStep) })).filter((item) => item.point);

  return (
    <div className="chart-wrap">
      <svg
        className="curve-chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${metricConfig[metric].label} over rollout updates`}
        onMouseLeave={() => setHoverStep(null)}
        onMouseMove={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          const relative = ((event.clientX - rect.left) / rect.width) * width;
          const step = ((relative - margin.left) / chartWidth) * targetStep;
          setHoverStep(Math.max(0, Math.min(targetStep, Math.round(step))));
        }}
      >
        <rect x={margin.left} y={margin.top} width={chartWidth} height={chartHeight} rx="6" className="plot-bg" />
        {ticksY.map((tick) => (
          <g key={tick}>
            <line x1={margin.left} x2={width - margin.right} y1={y(tick)} y2={y(tick)} className="grid-line" />
            <text x={margin.left - 13} y={y(tick) + 4} textAnchor="end" className="axis-label">{formatMetric(tick, metric, true)}</text>
          </g>
        ))}
        {ticksX.map((tick) => (
          <g key={tick}>
            <line x1={x(tick)} x2={x(tick)} y1={margin.top} y2={height - margin.bottom} className="grid-line vertical" />
            <text x={x(tick)} y={height - 19} textAnchor="middle" className="axis-label">{tick.toLocaleString()}</text>
          </g>
        ))}
        <text x={margin.left + chartWidth / 2} y={height - 2} textAnchor="middle" className="axis-title">RL update</text>
        {chartRuns.map(({ run, raw, smoothed }) => {
          const toPolyline = (values: { step: number; value: number }[]) => values.map((point) => `${x(point.step)},${y(point.value)}`).join(" ");
          return (
            <g key={run.id}>
              {smoothing > 1 && <polyline points={toPolyline(raw)} fill="none" stroke={run.color} className="raw-line" />}
              <polyline points={toPolyline(smoothed)} fill="none" stroke={run.color} className="metric-line" />
            </g>
          );
        })}
        {hoverStep !== null && (
          <line x1={x(hoverStep)} x2={x(hoverStep)} y1={margin.top} y2={height - margin.bottom} className="hover-line" />
        )}
        {tooltip.map(({ run, point }) => point && (
          <circle key={run.id} cx={x(point.step)} cy={y(point.value)} r="5" fill={run.color} className="hover-dot" />
        ))}
      </svg>
      {hoverStep !== null && tooltip.length > 0 && (
        <div className="chart-tooltip">
          <span className="tooltip-step">Near update {hoverStep.toLocaleString()}</span>
          {tooltip.map(({ run, point }) => point && (
            <span key={run.id} className="tooltip-row">
              <i style={{ background: run.color }} />
              <span>{run.shortLabel}</span>
              <strong>{formatMetric(point.value, metric)}</strong>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function StatusDot({ status }: { status: Run["status"] }) {
  return <span className={`status-dot ${status}`} aria-hidden="true" />;
}

export default function Dashboard() {
  const [data, setData] = useState<DashboardData>(fallbackData);
  const [metric, setMetric] = useState<Metric>("reward");
  const [smoothing, setSmoothing] = useState(25);
  const [visible, setVisible] = useState<Set<string>>(new Set());
  const [loaded, setLoaded] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const response = await fetch(`${metricsUrl}?t=${Date.now()}`, { cache: "no-store" });
      if (!response.ok) throw new Error("Metric feed unavailable");
      const payload = await response.json() as DashboardData;
      setData(payload);
      setVisible((current) => current.size ? current : new Set(payload.runs.map((run) => run.id)));
      setLoaded(true);
    } catch {
      setLoaded(true);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 30_000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const latestMean = useMemo(() => {
    const values = data.runs
      .map((run) => latestPointWithMetric(run.points, "reward")?.reward)
      .filter((value): value is number => value !== undefined);
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
  }, [data.runs]);

  const generated = new Date(data.generatedAt);
  const isFresh = loaded && Date.now() - generated.getTime() < 3 * 60_000;

  return (
    <main>
      <header className="topbar">
        <div className="brand-mark">CRL</div>
        <div className="brand-copy">
          <span className="eyebrow">PRETRAIN → SFT → RL COMPARISON</span>
          <h1>47.245M Qwen3: two checkpoints × two RL learning rates</h1>
        </div>
        <div className={`live-pill ${isFresh ? "fresh" : "stale"}`}>
          <span />
          {isFresh ? "LIVE" : "LAST SNAPSHOT"}
        </div>
      </header>

      <section className="summary-band">
        <div>
          <p className="kicker">{data.modelLabel}</p>
          <h2>Reward and pass@k curves</h2>
          <p className="lede">Each checkpoint is trained for 1,500 RL updates at 1e-5 or 1e-4 on {data.dataLabel} data. Each update uses 256 prompts × 8 samples.</p>
        </div>
        <div className="summary-stat">
          <span>Mean latest reward</span>
          <strong>{latestMean.toFixed(3)}</strong>
          <small>across active arms</small>
        </div>
      </section>

      <aside className="comparison-note" aria-label="Checkpoint configuration difference">
        <strong>Checkpoint configuration difference</strong>
        <span>{data.comparisonNote}</span>
      </aside>

      <section className="run-grid" aria-label="Run progress">
        {data.runs.map((run) => {
          const latestReward = latestPointWithMetric(run.points, "reward");
          const latestPass8 = latestPointWithMetric(run.points, "pass8");
          return (
            <article className="run-card" key={run.id} style={{ "--run-color": run.color } as React.CSSProperties}>
              <div className="run-card-head">
                <span className="run-swatch" />
                <div>
                  <h3>{run.shortLabel}</h3>
                  <p>{run.label}</p>
                </div>
                <span className="status-label"><StatusDot status={run.status} />{run.status}</span>
              </div>
              <div className="run-metrics">
                <div><span>Update</span><strong>{run.latestStep.toLocaleString()}</strong><small>/ 1,500</small></div>
                <div><span>Reward</span><strong>{latestReward?.reward?.toFixed(3) ?? "—"}</strong></div>
                <div><span>Pass@8</span><strong>{latestPass8?.pass8?.toFixed(3) ?? "—"}</strong></div>
              </div>
              <div className="progress-track"><span style={{ width: `${Math.min(100, (run.latestStep / data.targetStep) * 100)}%` }} /></div>
              <div className="checkpoint-note">Latest checkpoint: {run.checkpoint.toLocaleString()}</div>
            </article>
          );
        })}
        {!data.runs.length && loaded && <div className="empty-state">Waiting for the first Modal telemetry refresh…</div>}
      </section>

      <section className="chart-panel">
        <div className="chart-heading">
          <div>
            <p className="kicker">COMPARISON</p>
            <h2>{metricConfig[metric].label}</h2>
          </div>
          <div className="chart-controls">
            <label>
              Metric
              <select value={metric} onChange={(event) => setMetric(event.target.value as Metric)}>
                {Object.entries(metricConfig).map(([value, config]) => <option key={value} value={value}>{config.label}</option>)}
              </select>
            </label>
            <label>
              Curve
              <select value={smoothing} onChange={(event) => setSmoothing(Number(event.target.value))}>
                <option value={1}>Raw only</option>
                <option value={10}>10-update mean</option>
                <option value={25}>25-update mean</option>
                <option value={50}>50-update mean</option>
              </select>
            </label>
          </div>
        </div>

        <div className="legend" aria-label="Visible runs">
          {data.runs.map((run) => (
            <label key={run.id} className={visible.has(run.id) ? "selected" : ""}>
              <input
                type="checkbox"
                checked={visible.has(run.id)}
                onChange={() => setVisible((current) => {
                  const next = new Set(current);
                  if (next.has(run.id)) next.delete(run.id); else next.add(run.id);
                  return next;
                })}
              />
              <i style={{ background: run.color }} />
              {run.shortLabel}
            </label>
          ))}
        </div>

        <CurveChart runs={data.runs} targetStep={data.targetStep} metric={metric} smoothing={smoothing} visible={visible} />
        <div className="chart-footnote">
          <span><i className="sample-line raw" /> Raw per-update value</span>
          {smoothing > 1 && <span><i className="sample-line smooth" /> {smoothing}-update moving average</span>}
          <span className="sampling-note">{metricConfig[metric].note}</span>
        </div>
      </section>

      <footer>
        <span>Source: {data.source}</span>
        <span className="endpoint-note">Dashboard and telemetry hosted on Modal</span>
        <span>Updated {loaded && Number.isFinite(generated.getTime()) ? generated.toLocaleString() : "—"}</span>
        <button type="button" onClick={() => void refresh()}>Refresh now</button>
      </footer>
    </main>
  );
}
