import { execFile } from "node:child_process";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const outputPath = new URL("../public/data.json", import.meta.url);
const temporaryPath = new URL("../public/data.json.tmp", import.meta.url);
const watch = process.argv.includes("--watch");
const publishToModal = process.argv.includes("--publish-modal") || watch;
const targetStep = 1500;

const runSpecs = [
  {
    id: "pt10b-sft3-rl1e-5",
    label: "10,000,002,048 PT tokens → 77,717 SFT rows × 3 (1,388 updates, global batch 168)",
    shortLabel: "10.000B PT → full SFT ×3 · RL 1e-5",
    legacyRunId: "b4h-band-lr1e5-rl1500-r2",
    appId: "ap-yNc4nzNDavatRgris4O74j",
    callId: "fc-01KZT8YFHFDNHWVSG4VP56DAME",
    color: "#59a6ff",
  },
  {
    id: "pt10b-sft3-rl1e-4",
    label: "10,000,002,048 PT tokens → 77,717 SFT rows × 3 (1,388 updates, global batch 168)",
    shortLabel: "10.000B PT → full SFT ×3 · RL 1e-4",
    legacyRunId: "b4h-band-lr1e4-rl1500-r2",
    appId: "ap-ZobtENeJxDd5gG3b4LcMTQ",
    callId: "fc-01KZT8YFCRC2HKMC5Q4YYXHCHE",
    color: "#ffcc66",
  },
  {
    id: "c6p5e18-a0400-b0023-rl1e-5",
    label: "9,181,735,000 PT tokens → SFT × 3 (771 updates, effective batch 256; C=6.5e18, α=0.400, β=0.023)",
    shortLabel: "9.182B PT → SFT ×3 (α=.400, β=.023) · RL 1e-5",
    legacyRunId: "prev50m-a0400-band-lr1e5-rl1500-r2",
    appId: "ap-irBPsRpApz62Vdy8FCmkMY",
    callId: "fc-01KZT8YFD53N8XFDMHC2XHZ23V",
    color: "#7ee2b8",
  },
  {
    id: "c6p5e18-a0400-b0023-rl1e-4",
    label: "9,181,735,000 PT tokens → SFT × 3 (771 updates, effective batch 256; C=6.5e18, α=0.400, β=0.023)",
    shortLabel: "9.182B PT → SFT ×3 (α=.400, β=.023) · RL 1e-4",
    legacyRunId: "prev50m-a0400-band-lr1e4-rl1500-r2",
    appId: "ap-6n000VGJfR4PlLLemeovAY",
    callId: "fc-01KZT8YG5A78PK40G1RGWGES64",
    color: "#ff7a90",
  },
];

function stripAnsi(value) {
  return value.replace(/\u001b\[[0-9;]*m/g, "");
}

async function modal(args) {
  const { stdout } = await execFileAsync("modal", args, {
    maxBuffer: 64 * 1024 * 1024,
    timeout: 60_000,
  });
  return stripAnsi(stdout);
}

function parsePassrate(logs) {
  const byStep = new Map();
  for (const line of logs.split("\n")) {
    const stepMatch = line.match(/passrate\s+(\d+):\s+\{(.+)\}/);
    if (!stepMatch) continue;
    const values = {};
    const metricPattern = /passrate\/pass@(\d+)'\s*:\s*(?:np\.float64\()?([-+\d.eE]+)\)?/g;
    for (const match of stepMatch[2].matchAll(metricPattern)) {
      values[`pass${match[1]}`] = Number(match[2]);
    }
    if (Number.isFinite(values.pass1)) {
      byStep.set(Number(stepMatch[1]), {
        step: Number(stepMatch[1]),
        reward: values.pass1,
        ...values,
      });
    }
  }
  return [...byStep.values()].sort((a, b) => a.step - b.step);
}

const numericPattern = "[-+]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][-+]?\\d+)?";

function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function extractMetric(body, metric) {
  const pattern = new RegExp(
    `[\"']${escapeRegex(metric)}[\"']\\s*:\\s*(?:np\\.float64\\()?(${numericPattern})\\)?`,
  );
  const match = body.match(pattern);
  if (!match) return undefined;
  const value = Number(match[1]);
  return Number.isFinite(value) ? value : undefined;
}

function parseMetricSeries(logs, lineLabel, metrics) {
  const byStep = new Map();
  const linePattern = new RegExp(`${lineLabel}\\s+(\\d+):\\s+\\{(.+)\\}`);
  for (const line of logs.split("\n")) {
    const match = line.match(linePattern);
    if (!match) continue;
    const point = { step: Number(match[1]) };
    for (const [field, metric] of Object.entries(metrics)) {
      const value = extractMetric(match[2], metric);
      if (value !== undefined) point[field] = value;
    }
    if (Object.keys(point).length > 1) byStep.set(point.step, point);
  }
  return [...byStep.values()].sort((a, b) => a.step - b.step);
}

function mergeSeries(...series) {
  const byStep = new Map();
  for (const points of series) {
    for (const point of points) {
      byStep.set(point.step, { ...byStep.get(point.step), ...point });
    }
  }
  return [...byStep.values()].sort((a, b) => a.step - b.step);
}

function parseCheckpoint(logs) {
  let latest = 0;
  for (const match of logs.matchAll(/iter_(\d+)/g)) {
    latest = Math.max(latest, Number(match[1]));
  }
  return latest;
}

async function previousData() {
  try {
    return JSON.parse(await readFile(outputPath, "utf8"));
  } catch {
    return { runs: [] };
  }
}

async function collect() {
  const previous = await previousData();
  const previousById = new Map(previous.runs.map((run) => [run.id, run]));

  let appStates = new Map();
  try {
    const apps = JSON.parse(await modal(["app", "list", "--json"]));
    appStates = new Map(apps.map((app) => [app["App ID"], app]));
  } catch (error) {
    console.error(`Could not refresh app state: ${error.message}`);
  }

  const runs = await Promise.all(
    runSpecs.map(async (spec) => {
      const old = previousById.get(spec.id);
      let points = old?.points ?? [];
      let checkpoint = old?.checkpoint ?? 0;
      try {
        const [passrateLogs, entropyLogs, trainLogs, zeroGroupLogs, checkpointLogs] = await Promise.all([
          modal([
            "app",
            "logs",
            spec.appId,
            "--function-call",
            spec.callId,
            "--since",
            "24h",
            "--tail",
            "10000",
            "--search",
            "passrate/pass@1",
          ]),
          modal([
            "app",
            "logs",
            spec.appId,
            "--function-call",
            spec.callId,
            "--since",
            "24h",
            "--tail",
            "10000",
            "--search",
            "rollout/entropy",
          ]),
          modal([
            "app",
            "logs",
            spec.appId,
            "--function-call",
            spec.callId,
            "--since",
            "24h",
            "--tail",
            "10000",
            "--search",
            "train/grad_norm",
          ]),
          modal([
            "app",
            "logs",
            spec.appId,
            "--function-call",
            spec.callId,
            "--since",
            "24h",
            "--tail",
            "10000",
            "--search",
            "rollout/zero_std/all_zero_percentage",
          ]),
          modal([
            "app",
            "logs",
            spec.appId,
            "--function-call",
            spec.callId,
            "--since",
            "24h",
            "--tail",
            "1000",
            "--search",
            "Saved checkpoint",
          ]),
        ]);
        const passratePoints = parsePassrate(passrateLogs);
        const entropyPoints = parseMetricSeries(entropyLogs, "rollout", {
          entropy: "rollout/entropy",
        });
        const trainPoints = parseMetricSeries(trainLogs, "step", {
          gradNorm: "train/grad_norm",
          ppoKl: "train/ppo_kl",
        });
        const zeroGroupPoints = parseMetricSeries(zeroGroupLogs, "perf", {
          allZeroPercentage: "rollout/zero_std/all_zero_percentage",
        });
        points = mergeSeries(points, passratePoints, entropyPoints, trainPoints, zeroGroupPoints);
        checkpoint = Math.max(checkpoint, parseCheckpoint(checkpointLogs));
      } catch (error) {
        console.error(`${spec.shortLabel}: ${error.message}`);
      }

      const app = appStates.get(spec.appId);
      const latestStep = points.at(-1)?.step ?? 0;
      const running = app && Number(app.Tasks) > 0 && !String(app.State).includes("stopped");
      const status = checkpoint >= targetStep || latestStep >= targetStep - 1
        ? "complete"
        : running ? "running" : "stopped";
      return {
        ...spec,
        status,
        latestStep,
        checkpoint,
        points,
      };
    }),
  );

  const payload = {
    schemaVersion: 2,
    generatedAt: new Date().toISOString(),
    targetStep,
    refreshSeconds: 30,
    dataLabel: "mixed-outcome filtered (1–15/16)",
    modelLabel: "Qwen3 · 47,245,312 parameters",
    comparisonNote: "These checkpoints are not exact PT/SFT recipe matches. Pretraining differs in context handling, vocabulary, batch size, learning-rate schedule, and data selection; SFT differs in effective batch size, precision, weight decay, and exact rows. Within each checkpoint, the two RL learning-rate arms are matched.",
    source: "Modal application logs",
    runs,
  };

  await mkdir(new URL("../public/", import.meta.url), { recursive: true });
  await writeFile(temporaryPath, `${JSON.stringify(payload)}\n`, "utf8");
  await rename(temporaryPath, outputPath);
  if (publishToModal) {
    try {
      await execFileAsync(
        "modal",
        [
          "volume",
          "put",
          "chess-rl-dashboard-data",
          fileURLToPath(outputPath),
          "data.json",
          "--force",
        ],
        { maxBuffer: 4 * 1024 * 1024, timeout: 60_000 },
      );
    } catch (error) {
      console.error(`Could not publish dashboard snapshot: ${error.message}`);
    }
  }
  console.log(
    `[${payload.generatedAt}] ${runs.map((run) => `${run.shortLabel}=${run.latestStep}`).join(" · ")}`,
  );
}

await collect();
if (watch) {
  setInterval(() => collect().catch((error) => console.error(error)), 30_000);
}
