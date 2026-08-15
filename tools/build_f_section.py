"""Generate the expanded section F (table + 4 SVG curve panels) for
results_summary.html from the wave-2 measurement files."""
import json
import re
from pathlib import Path

S = Path(__file__).parent

RUNS = ["rollband-lr1e4-rl1500", "k16band-lr1e4-rl1500",
        "e2w1band-lr1e4-rl1500", "e3p2band-lr1e4-rl1500"]
KEY = {r: r.replace("-rl1500", "") for r in RUNS}
TAG = {"rollband-lr1e4-rl1500": "ROLLBAND", "k16band-lr1e4-rl1500": "K16BAND",
       "e2w1band-lr1e4-rl1500": "E2W1BAND", "e3p2band-lr1e4-rl1500": "E3P2BAND"}
LABEL = {
    "rollband-lr1e4-rl1500": "interleaved (rollout traces)",
    "k16band-lr1e4-rl1500": "interleaved (eval traces, k=16)",
    "e2w1band-lr1e4-rl1500": "pretrain-then-RL (10B, one run)",
    "e3p2band-lr1e4-rl1500": "pretrain-then-RL (10B, two runs)",
}
# fixed color assignment, same as the working dashboard's second-round panel
COLOR = {"k16band-lr1e4-rl1500": "--c1", "e2w1band-lr1e4-rl1500": "--c2",
         "e3p2band-lr1e4-rl1500": "--c3", "rollband-lr1e4-rl1500": "--c4"}

# step-0 anchors (full 53k evals + held-out loss of each base checkpoint)
BASE = {
    "rollband-lr1e4-rl1500": {"pass1": 0.384, "pass8": 0.565, "pass16": 0.610, "pt": 0.4831},
    "k16band-lr1e4-rl1500": {"pass1": 0.348, "pass8": 0.566, "pass16": 0.621, "pt": 0.4833},
    "e2w1band-lr1e4-rl1500": {"pass1": 0.148, "pass8": 0.488, "pass16": 0.601, "pt": 0.4823},
    "e3p2band-lr1e4-rl1500": {"pass1": 0.151, "pass8": 0.487, "pass16": 0.601, "pt": 0.4831},
}
FINAL = {
    "rollband-lr1e4-rl1500": {"pass1": 0.5139, "pass8": 0.6008, "pass16": 0.6204, "pt": 0.50106, "sft": 0.65198},
    "k16band-lr1e4-rl1500": {"pass1": 0.4938, "pass8": 0.5973, "pass16": 0.6204, "pt": 0.50091, "sft": 0.63177},
    "e2w1band-lr1e4-rl1500": {"pass1": 0.4296, "pass8": 0.5785, "pass16": 0.6147, "pt": 0.51036, "sft": 0.75235},
    "e3p2band-lr1e4-rl1500": {"pass1": 0.4006, "pass8": 0.5724, "pass16": 0.6153, "pt": 0.50518, "sft": 0.67022},
}


RESUMED = {"e2w1band-lr1e4-rl3000r": "e2w1band-lr1e4-rl1500",
           "e3p2band-lr1e4-rl3000r2": "e3p2band-lr1e4-rl1500"}


def load_reward() -> dict[str, list[dict]]:
    src = S / "wave2_curves_new.jsonl"
    if not src.exists() or src.stat().st_size == 0:
        src = S / "wave2_curves.jsonl"
    out = {}
    for line in src.read_text().splitlines():
        if not line.strip().startswith("{"):
            continue
        d = json.loads(line)
        if d["run"] in RUNS:
            out[d["run"]] = d["points"]
    ext = S / "rl3000r_curves.jsonl"
    if ext.exists():
        for line in ext.read_text().splitlines():
            if not line.strip().startswith("{"):
                continue
            d = json.loads(line)
            base = RESUMED.get(d["run"])
            if base in out:
                out[base] = out[base] + [p_ for p_ in d["points"]
                                         if p_["rollout"] >= 1500]
    return out


BASETAG = {"rollband-lr1e4-rl1500": "ROLLBASE", "k16band-lr1e4-rl1500": "K16BASE",
           "e2w1band-lr1e4-rl1500": "E2W1BASE", "e3p2band-lr1e4-rl1500": "E3P2BASE"}
RESUMED_LOSS = {"e2w1band-lr1e4-rl1500": "ptloss_e2w1band-lr1e4-rl3000r.out",
                "e3p2band-lr1e4-rl1500": "ptloss_e3p2band-lr1e4-rl3000r2.out"}


def load_base_sft() -> dict[str, float]:
    out = {}
    f = S / "ptloss_bases.out"
    if f.exists():
        for m in re.finditer(r'\{"tag": "[^"]*"[^}]*\}', f.read_text()):
            d = json.loads(m.group(0))
            for r, t in BASETAG.items():
                if d["tag"].startswith(t):
                    out[r] = d["sft_loss"]
    return out


def load_ptloss() -> tuple[dict[str, list[tuple[int, float]]], dict[str, list[tuple[int, float]]]]:
    pts = {r: [(0, BASE[r]["pt"])] for r in RUNS}
    base_sft = load_base_sft()
    spts = {r: ([(0, base_sft[r])] if r in base_sft else []) for r in RUNS}
    for r in RUNS:
        f = S / f"ptloss_{r}.out"
        if not f.exists():
            continue
        seen = set()
        for m in re.finditer(r'\{"tag": "[^"]*"[^}]*\}', f.read_text()):
            d = json.loads(m.group(0))
            step = int(d["tag"].split("@")[1])
            if step in seen:
                continue
            seen.add(step)
            pts[r].append((step, d["pt_heldout_loss"]))
            spts[r].append((step, d["sft_loss"]))
    for r, fname in RESUMED_LOSS.items():
        f = S / fname
        if not f.exists():
            continue
        seen = {st for st, _ in pts[r]}
        for m in re.finditer(r'\{"tag": "[^"]*"[^}]*\}', f.read_text()):
            d = json.loads(m.group(0))
            step = int(d["tag"].split("@")[1])
            if step in seen:
                continue
            seen.add(step)
            pts[r].append((step, d["pt_heldout_loss"]))
            spts[r].append((step, d["sft_loss"]))
    for r in RUNS:
        if not any(s == 1500 for s, _ in pts[r]):
            pts[r].append((1500, FINAL[r]["pt"]))
        if not any(s == 1500 for s, _ in spts[r]):
            spts[r].append((1500, FINAL[r]["sft"]))
        pts[r].sort()
        spts[r].sort()
    return pts, spts


RESUMED_PASS_KEY = {"e2w1band-lr1e4r": "e2w1band-lr1e4-rl1500",
                    "e3p2band-lr1e4r2": "e3p2band-lr1e4-rl1500"}


def load_pass() -> dict[str, list[dict]]:
    curve = json.loads((S / "rl_pass_curve_full.json").read_text())
    pts = {r: [] for r in RUNS}
    for row in curve:
        for r in RUNS:
            if row["run"] == KEY[r]:
                pts[r].append(row)
        if row["run"] in RESUMED_PASS_KEY:
            pts[RESUMED_PASS_KEY[row["run"]]].append(row)
    pts["e2w1band-lr1e4-rl1500"].append(
        {"step": 3000, "pass1": 0.485, "pass8": 0.5963, "pass16": 0.6227})
    pts["e3p2band-lr1e4-rl1500"].append(
        {"step": 3000, "pass1": 0.4537, "pass8": 0.5886, "pass16": 0.6211})
    for r in RUNS:
        pts[r].append({"step": 0, "pass1": BASE[r]["pass1"], "pass8": BASE[r]["pass8"],
                       "pass16": BASE[r]["pass16"]})
        pts[r].append({"step": 1500, "pass1": FINAL[r]["pass1"], "pass8": FINAL[r]["pass8"],
                       "pass16": FINAL[r]["pass16"]})
        pts[r].sort(key=lambda d: d["step"])
    return pts


def smooth(points, win=15, stride=10):
    vals = [(p["rollout"] + 1, p["mean_reward"]) for p in points]
    out = []
    for i in range(0, len(vals), stride):
        lo = max(0, i - win // 2)
        hi = min(len(vals), i + win // 2 + 1)
        out.append((vals[i][0], sum(v for _, v in vals[lo:hi]) / (hi - lo)))
    if vals and out[-1][0] != vals[-1][0]:
        out.append(vals[-1])
    return out


W, H = 460, 300
ML, MR, MT, MB = 46, 18, 14, 30


def panel(title, series, ymin, ymax, yticks, yfmt, markers, panel_id, tipfmt=None,
          xmax=1500):
    tipfmt = tipfmt or yfmt

    def sx(step):
        return ML + (W - ML - MR) * step / xmax

    def sy(v):
        return MT + (H - MT - MB) * (1 - (v - ymin) / (ymax - ymin))

    g = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}" data-panel="{panel_id}">']
    g.append(f'<text x="{ML}" y="{MT - 3}" class="ctitle">{title}</text>')
    for t in yticks:
        y = sy(t)
        g.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{W - MR}" y2="{y:.1f}" class="grid"/>')
        g.append(f'<text x="{ML - 5}" y="{y + 3.5:.1f}" class="tick" text-anchor="end">{yfmt(t)}</text>')
    for t in range(0, xmax + 1, 500 if xmax <= 1500 else 1000):
        g.append(f'<text x="{sx(t):.1f}" y="{H - 12}" class="tick" text-anchor="middle">{t}</text>')
    if xmax > 1500:
        g.append(f'<line x1="{sx(1500):.1f}" y1="{MT}" x2="{sx(1500):.1f}" y2="{H - MB}" '
                 f'class="grid" stroke-dasharray="4 4"/>')
    g.append(f'<text x="{(ML + W - MR) / 2:.0f}" y="{H - 1}" class="tick" text-anchor="middle">RL update</text>')
    for run in RUNS:
        pts = series[run]
        if not pts:
            continue
        col = f"var({COLOR[run]})"
        path = " ".join(f"{sx(s):.1f},{sy(v):.1f}" for s, v in pts)
        g.append(f'<polyline points="{path}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round"/>')
        for s, v in pts:
            tip = f"{LABEL[run]} — update {s}: {tipfmt(v)}"
            if markers:
                g.append(f'<circle cx="{sx(s):.1f}" cy="{sy(v):.1f}" r="3.5" fill="{col}" '
                         f'stroke="var(--panel)" stroke-width="1.5" data-tip="{tip}"/>')
            else:
                g.append(f'<circle cx="{sx(s):.1f}" cy="{sy(v):.1f}" r="6" fill="transparent" '
                         f'data-tip="{tip}"/>')
    g.append("</svg>")
    return "".join(g)


def main():
    reward = load_reward()
    ptloss, sftloss = load_ptloss()
    pk = load_pass()

    for r in RUNS:
        n = max((p["rollout"] for p in reward.get(r, [])), default=-1)
        print(f"{r}: reward to {n}, ptloss {len(ptloss[r])} pts, pass {len(pk[r])} pts")

    rew_series = {r: smooth(reward[r]) for r in RUNS}
    pt_series = {r: ptloss[r] for r in RUNS}
    sft_series = {r: sftloss[r] for r in RUNS}
    p1_series = {r: [(d["step"], d["pass1"]) for d in pk[r]] for r in RUNS}
    p16_series = {r: [(d["step"], d["pass16"]) for d in pk[r]] for r in RUNS}

    pct = lambda v: f"{v * 100:.0f}%"
    pct1 = lambda v: f"{v * 100:.1f}%" if isinstance(v, float) and v < 1 else str(v)
    loss3 = lambda v: f"{v:.3f}"

    panels = "\n".join([
        panel("Training reward (mean, own solvable-only set)", rew_series,
              0.2, 0.9, [0.2, 0.4, 0.6, 0.8], pct, False, "reward", lambda v: f"{v*100:.1f}%",
              xmax=3000),
        panel("Held-out pretraining loss (CE)", pt_series,
              0.478, 0.522, [0.48, 0.49, 0.50, 0.51, 0.52], loss3, True, "ptloss", lambda v: f"{v:.4f}",
              xmax=3000),
        panel("SFT loss (CE, supervised tokens of fixed SFT rows)", sft_series,
              0.48, 0.84, [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80], loss3, True, "sftloss", lambda v: f"{v:.4f}",
              xmax=3000),
        panel("pass@1 (full or stride-8 eval)", p1_series,
              0.10, 0.55, [0.1, 0.2, 0.3, 0.4, 0.5], pct, True, "pass1", lambda v: f"{v*100:.1f}%",
              xmax=3000),
        panel("pass@16 (full or stride-8 eval)", p16_series,
              0.56, 0.66, [0.56, 0.58, 0.60, 0.62, 0.64], pct, True, "pass16", lambda v: f"{v*100:.1f}%",
              xmax=3000),
    ])

    legend = "".join(
        f'<span class="lg"><span class="sw" style="background:var({COLOR[r]})"></span>{LABEL[r]}</span>'
        for r in RUNS)

    # format-collapse timing for prose
    for r in ("e2w1band-lr1e4-rl1500",):
        below = [p["rollout"] + 1 for p in reward.get(r, []) if p["format_rate"] < 0.95]
        if below:
            print(f"{r}: format first below 95% at update {below[0]}, "
                  f"final format {reward[r][-1]['format_rate']:.3f}")

    (S / "f_panels.html").write_text(
        f'<div class="legend">{legend}</div>\n<div class="charts">\n{panels}\n</div>\n')
    print("wrote f_panels.html")


if __name__ == "__main__":
    main()
