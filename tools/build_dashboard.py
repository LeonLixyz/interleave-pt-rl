"""Build dashboard/interleaved_dashboard.html from current run data.

Inputs (scratchpad): metrics_{p1w1,e2w1,e3p2}.jsonl, rl_curves.jsonl.
Eval numbers are the bos-corrected matrix (namespace ablation_pass16_clean_v2_bos).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

S = Path("/private/tmp/claude-501/-Users-leonli66-Desktop-Research-RL-Chess-RL/cf172269-6c26-448b-8a10-904c1e927526/scratchpad")
OUT = Path("/Users/leonli66/Desktop/Research/RL/Chess RL/dashboard/interleaved_dashboard.html")


def load_loss(name: str, max_pts: int = 250):
    path = S / f"metrics_{name}.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in open(path):
        d = json.loads(line)
        m = d["metrics"]
        sft = m.get("train/sft_token_loss")
        rows.append({
            "step": d["step"],
            "pt": round(m["train/pretrain_token_loss"], 4),
            "sft": round(sft, 4) if sft is not None else None,
            "uni": round(m["train/loss"], 4),
            "lr": m["train/lr"],
        })
    # Drop the first 200 warmup steps: the initial loss spike (4.6 -> ~0.6)
    # compresses the y-axis and hides the region that matters.
    rows = [r for r in rows if r["step"] >= 200] or rows
    stride = max(1, len(rows) // max_pts)
    out = rows[::stride]
    if out and out[-1]["step"] != rows[-1]["step"]:
        out.append(rows[-1])
    return out


loss_series = {
    "P1W1": load_loss("p1w1"),
    "E2W1": load_loss("e2w1"),
    "E3-P2": load_loss("e3p2"),
    "E1-U-P2": load_loss("e1up2"),
}

RUN_LABELS = {
    "e1-u-rl1500": "E1-U", "e1-d-rl1500": "E1-D",
    "e2-u-rl3000": "E2-U", "e2-d-rl3000": "E2-D",
    "e3-u-rl3000": "E3-U", "e3-d-rl3000": "E3-D",
    "e1-u-rl1500-leg2": "E1-U leg2",
    "b2h-u-rl1500": "B2H-U", "p1w1-band-rl1500": "P1W1-band", "p1w1-band-lr1e4-rl1500": "P1W1-band lr1e-4", "e1-d-rl1500-leg2": "E1-D leg2",
}
SIDE_LABELS = {"B2H-U", "P1W1-band", "P1W1-band lr1e-4"}
WAVE2_LABELS = {
    "k16band-lr1e4-rl1500": "interleaved (eval traces, k=16)",
    "e2w1band-lr1e4-rl1500": "pretrain-then-RL (10B, one run)",
    "e3p2band-lr1e4-rl1500": "pretrain-then-RL (10B, two runs)",
    "rollband-lr1e4-rl1500": "interleaved (rollout traces)",
}
# controls resumed from the 1,500-step checkpoints: stitch each curve from the
# original run (updates 0-1499) + the resumed run (1500+)
W2CTL_LABELS = {
    "e2w1band-lr1e4-rl3000r": "pretrain-then-RL (one run), 3,000 updates",
    "e3p2band-lr1e4-rl3000r2": "pretrain-then-RL (two runs), 3,000 updates",
}
W2CTL_FIRST_HALF = {
    "e2w1band-lr1e4-rl3000r": "e2w1band-lr1e4-rl1500",
    "e3p2band-lr1e4-rl3000r2": "e3p2band-lr1e4-rl1500",
}
wave2_series = {}
w2ctl_series = {}
import json as _json
_w2_raw = {}
for _fname, _labels, _target in (
    ("wave2_curves_new.jsonl", WAVE2_LABELS, wave2_series),
    ("wave2_curves.jsonl", WAVE2_LABELS, wave2_series),
    ("rl3000r_curves.jsonl", W2CTL_LABELS, w2ctl_series),
):
    _f = S / _fname
    if not _f.exists():
        continue
    for _line in open(_f):
        if not _line.strip().startswith("{"):
            continue
        _d = _json.loads(_line)
        if _d["run"] in WAVE2_LABELS or _d["run"] in W2CTL_LABELS:
            _w2_raw.setdefault(_d["run"], _d["points"])
        if _d["run"] in _labels and _labels[_d["run"]] not in _target:
            _pts = _d["points"]
            if _d["run"] in W2CTL_FIRST_HALF:
                _first = _w2_raw.get(W2CTL_FIRST_HALF[_d["run"]], [])
                _pts = [p_ for p_ in _first if p_["rollout"] < 1500] + _pts
            _pts = [{"rollout": p_["rollout"], "reward": round(p_["mean_reward"], 4),
                     "fmt": round(p_["format_rate"], 4)} for p_ in _pts]
            _stride = max(1, len(_pts) // 300)
            _target[_labels[_d["run"]]] = _pts[::_stride]

rl_series = {}
rl3000_series = {}
side_series = {}
for line in open(S / "rl_curves.jsonl"):
    d = json.loads(line)
    label = RUN_LABELS[d["run"]]
    pts = [
        {"rollout": p["rollout"], "reward": round(p["mean_reward"], 4),
         "fmt": round(p["format_rate"], 4)} for p in d["points"]
    ]
    stride = max(1, len(pts) // 300)
    pts = pts[::stride] + ([pts[-1]] if stride > 1 and pts else [])
    target = (side_series if label in SIDE_LABELS
              else rl_series if label.startswith("E1") else rl3000_series)
    target[label] = pts

EVAL = [
    # arm, chain description, pass1, pass8, pass16, format
    ("P1W1", "5B-pretrain (+38.8k SFT mixed)", 0.1015, 0.3931, 0.5073, 0.9448),
    ("E2W1", "10B-pretrain, one run (+77.7k SFT mixed)", 0.1480, 0.4880, 0.6005, 0.9736),
    ("E3P2", "10B-pretrain, two runs — control for the chains below", 0.1507, 0.4869, 0.5996, 0.9614),
    ("B2H", "5B-pretrain (no SFT mixed) \u2192 SFT-stage (77.7k \u00d7 3 epochs)", 0.1846, 0.5301, 0.6319, 0.9696),
    ("E1U-RL", "5B-pretrain \u2192 RL(all, 1e-5, 1500)", 0.1379, 0.4179, 0.5159, 0.9765),
    ("E1D-RL", "5B-pretrain \u2192 RL(filter, 1e-5, 1500)", 0.1467, 0.4238, 0.5185, 0.9805),
    ("band-lr1e5", "5B-pretrain \u2192 RL(solvable-only, 1e-5, 1500)", 0.1455, 0.4146, 0.5084, 0.9808),
    ("band-lr1e4", "5B-pretrain \u2192 RL(solvable-only, 1e-4, 1500)", 0.3630, 0.5107, 0.5481, 0.9982),
    ("B2H-U", "5B-pretrain (no SFT mixed) \u2192 SFT-stage \u2192 RL(all, 1e-5, 1500)", 0.2267, 0.5395, 0.6273, 0.9677),
    ("E2-U", "10B-pretrain \u2192 RL(all, 1e-5, 3000)", 0.1919, 0.4998, 0.5939, 0.9856),
    ("E2-D", "10B-pretrain \u2192 RL(filter, 1e-5, 3000)", 0.1995, 0.5024, 0.5921, 0.9871),
    ("E3-U", "10B-pretrain(2 runs) \u2192 RL(all, 1e-5, 3000)", 0.1925, 0.5037, 0.5999, 0.9822),
    ("E3-D", "10B-pretrain(2 runs) \u2192 RL(filter, 1e-5, 3000)", 0.2004, 0.5077, 0.6021, 0.9843),
    ("E1UP2", "pretrain \u2192 RL(U) \u2192 pretrain: weights wash out", 0.1439, 0.4836, 0.5978, 0.9613),
    ("E1DP2", "pretrain \u2192 RL(D) \u2192 pretrain: weights wash out", 0.1509, 0.4868, 0.5992, 0.9607),
    ("LR4P2", "pretrain \u2192 RL(36.3% model) \u2192 pretrain: still washes out", 0.1478, 0.4886, 0.6016, 0.9616),
    ("TRACE-k1", "pretrain \u2192 RL \u2192 traces(29k) \u2192 pretrain", 0.2155, 0.5393, 0.6338, 0.9768),
    ("TRACE-k2", "pretrain \u2192 RL \u2192 traces(55k) \u2192 pretrain", 0.2533, 0.5568, 0.6398, 0.9838),
    ("TRACE-k4", "pretrain \u2192 RL \u2192 traces(103k) \u2192 pretrain — best coverage", 0.2916, 0.5687, 0.6413, 0.9889),
    ("TRACE-k8", "pretrain \u2192 RL \u2192 traces(188k) \u2192 pretrain", 0.3267, 0.5691, 0.6313, 0.9928),
    ("TRACE-k16", "pretrain \u2192 RL \u2192 traces(308k) \u2192 pretrain", 0.3483, 0.5659, 0.6214, 0.9945),
    ("E1-U final", "pretrain \u2192 RL-U \u2192 pretrain \u2192 RL-U (interleaved)", 0.1770, 0.4914, 0.5902, 0.9804),
    ("E1-D final", "pretrain \u2192 RL-D \u2192 pretrain \u2192 RL-D (interleaved)", 0.1924, 0.5018, 0.5979, 0.9818),
    ("INT-ROLL", "interleaved (rollout traces): 5B \u2192 RL \u2192 rollout traces \u2192 5B \u2192 RL \u2014 best in study", 0.5139, 0.6008, 0.6204, 0.9993),
    ("INT-K16", "interleaved (eval traces, k=16): 5B \u2192 RL \u2192 eval traces \u2192 5B \u2192 RL", 0.4938, 0.5973, 0.6204, 0.9993),
    ("PTRL-1", "pretrain-then-RL: 10B (one run) \u2192 RL(solvable-only, 1e-4, 1500)", 0.4296, 0.5785, 0.6147, 0.8665),
    ("PTRL-2", "pretrain-then-RL: 10B (two runs) \u2192 RL(solvable-only, 1e-4, 1500)", 0.4006, 0.5724, 0.6153, 0.9986),
    ("LOOP-1 base", "10B \u2192 RL \u2192 rollout traces \u2192 5B fresh pretrain (15B tokens total)", 0.4273, 0.6106, 0.6522, 0.9968),
    ("LOOP-2 base", "10B (two runs) \u2192 RL \u2192 rollout traces \u2192 5B fresh pretrain", 0.4246, 0.6090, 0.6529, 0.9977),
    ("LOOP-1", "10B loop (one run) + second RL round \u2014 best pass@1 and pass@16 in study", 0.5461, 0.6370, 0.6568, 0.9056),
    ("LOOP-2", "10B loop (two runs) + second RL round \u2014 best with format intact", 0.5293, 0.6311, 0.6525, 0.9993),
    ("PTRL-1 @3000", "pretrain-then-RL (10B, one run) at 3,000 updates \u2014 equal total RL to the interleaved chains", 0.4850, 0.5963, 0.6227, 0.8714),
    ("PTRL-2 @3000", "pretrain-then-RL (10B, two runs) at 3,000 updates \u2014 equal total RL", 0.4537, 0.5886, 0.6211, 0.9990),
]

_steps = json.loads((S / "rl_steps.json").read_text()) if (S / "rl_steps.json").exists() else {}

def _live(run: str, target: int) -> str:
    s = _steps.get(run)
    return f"step {s:,} / {target:,}" if s else "starting"

STATUS = [
    ("P1 root (P1W1)", "9,920 steps, v2r1 P1 leg, w=1.0", "done", "pass@1 10.2%, format 94.5%"),
    ("E2 root (E2W1)", "19,840 steps, monolithic P1+P2", "done", "pass@1 14.8%, format 97.4%"),
    ("E1-U first RL leg", "GRPO 1,500 updates, unfiltered, from P1", "done", "pass@1 13.8% (P1 was 10.2%)"),
    ("E1-U-P2", "9,920 steps, P2 leg, fresh optimizer, init from RL-1500", "done", "pass@1 14.4% ≈ E3-P2's 15.1%: midpoint RL washed out"),
    ("E1-U second RL leg", "GRPO 1,500 updates, seed 43, from E1-U-P2", "done", "pass@1 17.7% — 1.5 pts below E2-U/E3-U at equal budget"),
    ("E1-D first RL leg", "GRPO 1,500 updates, dynamic filter, from P1", "done", "pass@1 14.7% (U leg: 13.8%)"),
    ("E1-D-P2", "9,920 steps, P2 leg, fresh optimizer, init from RL-D-1500", "done", "pass@1 15.1% = E3-P2 exactly: wash-out replicated"),
    ("E1-D second RL leg", "GRPO 1,500 updates, seed 43, dynamic filter", "done", "final 19.2% vs E2-D/E3-D 20.0%: interleaving loses in D too"),
    ("E3-P2", "9,920 steps, P2 leg, fresh optimizer, init from P1", "done", "pass@1 15.1%, format 96.1%"),
    ("E2-U RL-3000", "GRPO 3,000 updates from E2W1", "done", "pass@1 19.2%, pass@16 59.4% (root 60.1%)"),
    ("E2-D RL-3000", "GRPO 3,000 updates from E2W1, dynamic filter", "done", "pass@1 20.0%"),
    ("E3-U RL-3000", "GRPO 3,000 updates from E3-P2", "done", "pass@1 19.3%, format 98.2%"),
    ("E3-D RL-3000", "GRPO 3,000 updates from E3-P2, dynamic filter", "done", "pass@1 20.0% = E2-D exactly"),
    ("Trace-transfer P2 (k=1)", "P2 leg + 29,112 verified RL traces in the stream, from P1", "done", "pass@1 21.6%, pass@16 63.4% — data transfer works"),
    ("Trace dosage sweep (k=2,4,8,16)", "same recipe with up to k traces per prompt", "done", "pass@1 scales to 34.8%; pass@16 peaks at k=4 (64.1%)"),
    ("E1 P2 stages + 2nd RL legs", "after first RL legs finish", "pending", ""),
    ("B2H-U RL", "GRPO 1,500 updates from B2H (staged full-SFT 3-epoch, 18.5%)", "done", "pass@1 22.7% — best lr 1e-5 model"),
    ("P1W1-band RL", "GRPO 1,500 updates from P1W1 on 1-15-wins filtered prompts (26,967)", "done", "pass@1 14.6% — filtering alone adds little"),
    ("P1W1-band lr 1e-4", "same filtered set, 10x learning rate", "done", "pass@1 36.3% — new best by far"),
    ("Second-stage PT from the 36.3% model", "9,920 steps, P2 leg, fresh optimizer", "done", "pass@1 14.8% = control: wash-out is total, dose-independent"),
    ("Second-round RL (4 runs)", "lr 1e-4, 1,500 updates, each on its own solvable-only set", "done", "interleaved (rollout traces) 51.4% > interleaved (eval traces) 49.4% > pretrain-then-RL 43.0/40.1%"),
    ("Equal-total-RL controls", "both pretrain-then-RL runs extended to 3,000 updates", "done", "48.5% / 45.4% vs interleaved 51.4% at equal budget \u2014 extra RL closed half the gap, repaired neither format nor LM damage"),
    ("10B loop arms", "harvest rollouts of both 10B RL runs \u2192 fresh-5B pretraining leg with traces \u2192 second RL round", "done", "new records: 54.6% / 52.9% pass@1, 65.7% pass@16; trace stage alone hit 42.7% with best-in-study PT loss"),
    ("Two-run 3,000-update control", "relaunched after a scoring-environment failure at ~update 1,950 damaged the first attempt", "running", "position-exact resume from the 1,500-update checkpoint"),
]

EVLOSS = [
    {"tag": "P1W1", "pt": 0.50104, "sft_p1": 0.52827, "sft_p2": 0.53915,
     "uniform": 0.50121, "p2_unseen": True},
    {"tag": "E2W1", "pt": 0.48225, "sft_p1": 0.50436, "sft_p2": 0.50613,
     "uniform": 0.48237, "p2_unseen": False},
    {"tag": "E3-P2", "pt": 0.48313, "sft_p1": 0.50679, "sft_p2": 0.50430,
     "uniform": 0.48325, "p2_unseen": False},
]

CURVE_LABELS = {"e1-u": "E1-U", "e1-d": "E1-D", "e2-u": "E2-U",
                "e2-d": "E2-D", "e3-u": "E3-U", "e3-d": "E3-D",
                "e1-u-leg2": "E1-U leg2", "b2h-u": "B2H-U", "p1w1-band": "P1W1-band", "p1w1-band-lr1e4": "P1W1-band lr1e-4", "e1-d-leg2": "E1-D leg2"}
W2_CURVE_LABELS = {
    "k16band-lr1e4": "interleaved (eval traces, k=16)",
    "rollband-lr1e4": "interleaved (rollout traces)",
    "e2w1band-lr1e4": "pretrain-then-RL (10B, one run)",
    "e3p2band-lr1e4": "pretrain-then-RL (10B, two runs)",
    "e2w1band-lr1e4r": "pretrain-then-RL (10B, one run)",
    "e3p2band-lr1e4r2": "pretrain-then-RL (10B, two runs)",
}
passcurve = {}
passcurve3000 = {}
passcurve_side = {}
passcurve_w2 = {}
raw = []
if (S / "rl_pass_curve_full.json").exists():
    raw += json.loads((S / "rl_pass_curve_full.json").read_text())
# step-0 baselines measured on the same stride-8 subset
raw += [
    {"run": "e1-u", "step": 0, "pass1": 0.1031, "pass16": 0.5150, "format": 0.9456},
    {"run": "e1-d", "step": 0, "pass1": 0.1031, "pass16": 0.5150, "format": 0.9456},
]
# full-eval anchors for the second-round runs (step 0 = base, 1500 = endpoint)
raw += [
    {"run": "rollband-lr1e4", "step": 0, "pass1": 0.384, "pass16": 0.610, "format": 0.9970},
    {"run": "rollband-lr1e4", "step": 1500, "pass1": 0.5139, "pass16": 0.6204, "format": 0.9993},
    {"run": "k16band-lr1e4", "step": 0, "pass1": 0.348, "pass16": 0.621, "format": 0.9945},
    {"run": "k16band-lr1e4", "step": 1500, "pass1": 0.4938, "pass16": 0.6204, "format": 0.9993},
    {"run": "e2w1band-lr1e4", "step": 0, "pass1": 0.148, "pass16": 0.601, "format": 0.9736},
    {"run": "e2w1band-lr1e4", "step": 1500, "pass1": 0.4296, "pass16": 0.6147, "format": 0.8665},
    {"run": "e3p2band-lr1e4", "step": 0, "pass1": 0.151, "pass16": 0.600, "format": 0.9614},
    {"run": "e3p2band-lr1e4", "step": 1500, "pass1": 0.4006, "pass16": 0.6153, "format": 0.9986},
]
for r in raw:
    if r["run"] in W2_CURVE_LABELS:
        passcurve_w2.setdefault(W2_CURVE_LABELS[r["run"]], []).append(
            {"step": r["step"], "p1": r["pass1"], "p16": r["pass16"], "fmt": r["format"]})
        continue
    label = CURVE_LABELS.get(r["run"])
    if label is None:
        continue
    target = (passcurve_side if label in SIDE_LABELS
              else passcurve if label.startswith("E1") else passcurve3000)
    step = r["step"] + (1500 if r["run"].endswith("-leg2") else 0)
    target.setdefault(label, []).append(
        {"step": step, "p1": r["pass1"], "p16": r["pass16"], "fmt": r["format"]})
for d in (passcurve, passcurve3000, passcurve_side, passcurve_w2):
    for v in d.values():
        v.sort(key=lambda x: x["step"])

data = {
    "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    "passcurve": passcurve,
    "passcurve3000": passcurve3000,
    "passcurve_side": passcurve_side,
    "evloss": EVLOSS,
    "loss": loss_series,
    "rl": rl_series,
    "rl3000": rl3000_series,
    "rlside": side_series,
    "rlwave2": wave2_series,
    "rlw2ctl": w2ctl_series,
    "passcurve_w2": passcurve_w2,
    "eval": [
        {"arm": a, "desc": d, "p1": p1, "p8": p8, "p16": p16, "fmt": f}
        for a, d, p1, p8, p16, f in EVAL
    ],
    "status": [
        {"name": n, "desc": d, "state": s, "note": x} for n, d, s, x in STATUS
    ],
}

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Interleaved Pretrain ↔ RL — 50M</title>
<style>
:root {
  --bg: #fcfcfb; --panel: #ffffff; --border: #d9d9d4;
  --ink: #1e1e1c; --ink2: #55554f; --muted: #8a8a82;
  --c1: #3b5fc0; --c2: #b26a00; --c3: #c22f88; --c4: #0e8a43; --c5: #7048d8;
  --good: #0e8a43; --run: #b26a00; --pend: #8a8a82;
  --grid: #ecece7;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #161615; --panel: #1e1e1c; --border: #3a3a36;
    --ink: #e8e8e3; --ink2: #b5b5ac; --muted: #7d7d74;
    --c1: #5b7fe0; --c2: #c07f14; --c3: #d5539f; --c4: #22a55c; --c5: #8a68e8;
    --good: #22a55c; --run: #c07f14; --pend: #7d7d74;
    --grid: #2c2c29;
  }
}
* { box-sizing: border-box; margin: 0; }
body {
  background: var(--bg); color: var(--ink);
  font: 14px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  padding: 24px; max-width: 1180px; margin: 0 auto;
}
h1 { font-size: 21px; font-weight: 650; }
h2 { font-size: 15px; font-weight: 650; margin: 0 0 2px; }
.sub { color: var(--ink2); font-size: 13px; margin-top: 2px; }
.gen { color: var(--muted); font-size: 12px; }
header { display: flex; justify-content: space-between; align-items: baseline;
         flex-wrap: wrap; gap: 8px; margin-bottom: 18px; }
section { background: var(--panel); border: 1px solid var(--border);
          border-radius: 8px; padding: 16px 18px; margin-bottom: 16px; }
.note { color: var(--ink2); font-size: 12.5px; margin-top: 6px; }
table { border-collapse: collapse; width: 100%; margin-top: 10px;
        font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border);
         font-size: 13px; }
th { color: var(--ink2); font-weight: 600; font-size: 12px;
     text-transform: uppercase; letter-spacing: 0.04em; }
td.num, th.num { text-align: right; }
.pill { display: inline-block; padding: 1px 9px; border-radius: 999px;
        font-size: 12px; font-weight: 600; }
.pill.done { color: var(--good); border: 1px solid var(--good); }
.pill.running { color: var(--run); border: 1px solid var(--run); }
.pill.pending { color: var(--pend); border: 1px solid var(--pend); }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
          gap: 16px; }
.chart-box { background: var(--panel); border: 1px solid var(--border);
             border-radius: 8px; padding: 14px 16px; }
svg { display: block; width: 100%; height: auto; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 6px;
          font-size: 12.5px; color: var(--ink2); }
.legend span::before { content: ""; display: inline-block; width: 10px; height: 10px;
                       border-radius: 2px; margin-right: 5px; vertical-align: -1px;
                       background: var(--sw); }
.tip { position: fixed; pointer-events: none; background: var(--panel);
       border: 1px solid var(--border); border-radius: 6px; padding: 5px 9px;
       font-size: 12px; color: var(--ink); display: none; z-index: 10;
       box-shadow: 0 2px 8px rgba(0,0,0,0.12); white-space: nowrap; }
.bar-row td { border-bottom: none; padding: 3px 10px; }
.barwrap { background: var(--grid); border-radius: 3px; height: 14px; position: relative; }
.bar { height: 14px; border-radius: 3px; position: absolute; left: 0; top: 0; }
.overflow { overflow-x: auto; }
.pipe { display: flex; align-items: center; flex-wrap: wrap; gap: 0; margin: 7px 0; }
.pipe .who { width: 240px; font-size: 13px; color: var(--ink); flex-shrink: 0; }
.pipe .who b { display: block; }
.pipe .who span { color: var(--ink2); font-size: 12px; }
.stage { border: 1.5px solid var(--muted); border-radius: 6px; padding: 3px 10px;
         font-size: 12.5px; white-space: nowrap; color: var(--ink); background: var(--panel); }
.stage.pt { border-color: var(--c1); }
.stage.rl { border-color: var(--c4); background: color-mix(in srgb, var(--c4) 10%, var(--panel)); }
.conn { width: 18px; height: 1.5px; background: var(--muted); flex-shrink: 0; }
.deflist { margin: 8px 0 0; font-size: 13px; color: var(--ink); }
.deflist dt { font-weight: 650; float: left; clear: left; width: 52px; }
.deflist dd { margin: 0 0 6px 62px; color: var(--ink2); }
</style>
</head>
<body>
<header>
  <div>
    <h1>Interleaved Pretrain ↔ RL — 50M (47.245M) Qwen3</h1>
    <div class="sub">All numbers on this page use the corrected evaluation and rollout code:
      every prompt starts with <code>&lt;bos&gt;</code>, matching training. Nothing here predates that fix.</div>
  </div>
  <div class="gen">updated __GENERATED__</div>
</header>

<section>
  <h2>What the names mean</h2>
  <dl class="deflist">
    <dt>P1</dt><dd>first half of pretraining: 5B tokens + half of the SFT rows (38,858), mixed into one shuffled stream, 9,920 steps.</dd>
    <dt>P2</dt><dd>second half: the <i>other</i> 5B tokens + the other half of the SFT rows. Same recipe, fresh optimizer and schedule.</dd>
    <dt>RL</dt><dd>GRPO on chess puzzles, 256 prompts × 8 samples per update, lr 1e-5.</dd>
    <dt>U / D</dt><dd>U = RL trains on every sampled prompt group. D = keeps only groups where some of the 8 samples succeed and some fail (mixed outcomes carry gradient signal); fully-failed and fully-solved groups are redrawn.</dd>
  </dl>
  <div class="pipe"><div class="who"><b>E1 — interleaved</b><span>does RL in the middle help later training?</span></div>
    <span class="stage pt">P1</span><span class="conn"></span><span class="stage rl">RL 1,500</span><span class="conn"></span><span class="stage pt">P2</span><span class="conn"></span><span class="stage rl">RL 1,500</span></div>
  <div class="pipe"><div class="who"><b>E2 — monolithic control</b><span>no midpoint at all</span></div>
    <span class="stage pt">P1 + P2 as one uninterrupted run</span><span class="conn"></span><span class="stage rl">RL 3,000</span></div>
  <div class="pipe"><div class="who"><b>E3 — split control</b><span>same midpoint as E1, but no RL there</span></div>
    <span class="stage pt">P1</span><span class="conn"></span><span class="stage pt">P2</span><span class="conn"></span><span class="stage rl">RL 3,000</span></div>
  <div class="note">All three see exactly the same pretraining+SFT tokens in exactly the same order — E2's single stream is literally E1/E3's P1 followed by P2. The only differences are the optimizer reset at the midpoint (E1, E3) and whether RL happens there (E1). Every arm ends with 3,000 total RL updates. Comparing E1 vs E3 isolates "RL in the middle"; E2 vs E3 isolates the schedule split.</div>
</section>

<section>
  <h2>Experiment status</h2>
  <div class="sub">Controlled matrix: E1 = P1 → RL 1500 → P2 → RL 1500 · E2 = monolithic P1+P2 → RL 3000 · E3 = P1 → fresh optimizer → P2 → RL 3000. Each × {U unfiltered, D dynamic filter}.</div>
  <div class="overflow"><table id="status-table"></table></div>
</section>

<section>
  <h2>Endpoint evaluations — full 53,157-prompt set, n = 16, temperature 1.0</h2>
  <div class="sub">pass@k via the unbiased estimator; format = format-valid rate.</div>
  <div class="overflow"><table id="eval-table"></table></div>
  <div class="note">A1/A4/B2H/B4H are current (bos-corrected) reference points from the SFT-injection ablation, shown for comparison. </div>
</section>

<div class="charts">
  <div class="chart-box">
    <h2>Pretraining loss (per-token CE on pretraining tokens, from step 200)</h2>
    <div id="chart-pt"></div>
    <div class="legend" id="leg-pt"></div>
  </div>
  <div class="chart-box">
    <h2>SFT loss (per-token CE on supervised SFT tokens, from step 200)</h2>
    <div id="chart-sft"></div>
    <div class="legend" id="leg-sft"></div>
  </div>
  <div class="chart-box">
    <h2>Combined training loss (all supervised tokens, from step 200)</h2>
    <div id="chart-uni"></div>
    <div class="legend" id="leg-uni"></div>
    <div class="note">At weight 1.0 this is the plain per-token mean over pretraining + SFT tokens together — the actual training objective. It tracks the pretraining curve closely because SFT tokens are ~0.5% of the stream.</div>
  </div>
  <div class="chart-box">
    <h2>Evaluation losses at stage endpoints</h2>
    <div class="overflow"><table id="evloss-table"></table></div>
    <div class="note">Held-out PT: 2,048 packed 3,072-token windows (6.3M tokens) from shards the frozen 10B training selection never used. SFT halves: 2,048-row probes per half; for a P1-stage model the P2 half is genuinely unseen, for full-exposure models both halves were seen exactly once. Uniform = PT and SFT means combined at the training mixture share (0.52% SFT).</div>
  </div>
  <div class="chart-box">
    <h2>E1 RL legs: pass@1 vs effective RL step (leg 2 offset by +1,500)</h2>
    <div id="chart-p1c"></div>
    <div class="legend" id="leg-p1c"></div>
    <div class="note">Checkpoints converted to HF and scored with the same evaluator as the endpoint table. pass@16 stays flat (~51.5%) while pass@1 climbs — RL is sharpening the policy, not expanding coverage.</div>
  </div>
  <div class="chart-box">
    <h2>E1 RL legs: pass@16 vs effective RL step (same subset)</h2>
    <div id="chart-p16c"></div>
    <div class="legend" id="leg-p16c"></div>
  </div>
  <div class="chart-box">
    <h2>Second-round RL (lr 1e-4, own solvable-only set): mean reward per update</h2>
    <div id="chart-rww2"></div>
    <div class="legend" id="leg-rww2"></div>
    <div class="note">Rewards are measured on each run's own solvable-only distribution — levels are comparable in trend, not absolute value, across runs.</div>
  </div>
  <div class="chart-box">
    <h2>Second-round RL: pass@1 vs RL step (stride-8 evals every 200; full evals at 0 and 1,500)</h2>
    <div id="chart-p1w2"></div>
    <div class="legend" id="leg-p1w2"></div>
    <div class="note">The interleaved runs start 20+ points ahead and are never caught; ordering is fixed from step 200.</div>
  </div>
  <div class="chart-box">
    <h2>Second-round RL: pass@16 vs RL step (same points)</h2>
    <div id="chart-p16w2"></div>
    <div class="legend" id="leg-p16w2"></div>
    <div class="note">Pretrain-then-RL (one run) dips to 57.4% at step 200 and needs ~1,200 updates to recover its starting coverage; the interleaved runs never dip.</div>
  </div>
  <div class="chart-box">
    <h2>Equal-total-RL controls (3,000 updates, training): mean reward per update</h2>
    <div id="chart-rww2c"></div>
    <div class="legend" id="leg-rww2c"></div>
    <div class="note">Both pretrain-then-RL runs extended to 3,000 updates to match the interleaved runs' total RL budget (1,500 + 1,500): resumed from the 1,500-step checkpoint with full optimizer state, same solvable-only sets and recipe, constant lr 1e-4. Updates 0–1,499 are the original runs' curve.</div>
  </div>
  <div class="chart-box">
    <h2>Side runs: pass@1 vs RL step (B2H base / filtered data / 10× lr)</h2>
    <div id="chart-p1side"></div>
    <div class="legend" id="leg-p1side"></div>
    <div class="note">All from 5B-class bases at 1,500-update targets. P1W1-band trains on the 26,967 prompts P1W1 solves 1–15 times of 16; the lr1e-4 variant is the same data at 10× learning rate.</div>
  </div>
  <div class="chart-box">
    <h2>Side runs: pass@16 vs RL step (same subset)</h2>
    <div id="chart-p16side"></div>
    <div class="legend" id="leg-p16side"></div>
  </div>
  <div class="chart-box">
    <h2>Side runs: mean reward per rollout update</h2>
    <div id="chart-rwside"></div>
    <div class="legend" id="leg-rwside"></div>
    <div class="note">Band-run rewards are measured on their filtered training distribution — levels aren't comparable to full-data runs; the pass@k charts above are the like-for-like comparison.</div>
  </div>
  <div class="chart-box">
    <h2>E2/E3 final RL: pass@1 vs RL step (evaluated checkpoints, same subset)</h2>
    <div id="chart-p1c3"></div>
    <div class="legend" id="leg-p1c3"></div>
  </div>
  <div class="chart-box">
    <h2>E2/E3 final RL: pass@16 vs RL step (same subset)</h2>
    <div id="chart-p16c3"></div>
    <div class="legend" id="leg-p16c3"></div>
  </div>
  <div class="chart-box">
    <h2>E1 first RL legs (from P1): mean reward per update, 2,048 trajectories each</h2>
    <div id="chart-rw"></div>
    <div class="legend" id="leg-rw"></div>
    <div class="note">E1-D's curve is measured on its accepted (nonzero-variance) batches, so it is not directly comparable to E1-U's unfiltered mean.</div>
  </div>
  <div class="chart-box">
    <h2>E1 first RL legs: format-valid rate per update</h2>
    <div id="chart-fmt"></div>
    <div class="legend" id="leg-fmt"></div>
  </div>
  <div class="chart-box">
    <h2>E2/E3 final RL (3,000 updates): mean reward per update</h2>
    <div id="chart-rw3"></div>
    <div class="legend" id="leg-rw3"></div>
    <div class="note">E2 arms start from the monolithic root (E2W1), E3 arms from the two-cosine root (E3-P2). D arms are measured on accepted nonzero-variance batches.</div>
  </div>
  <div class="chart-box">
    <h2>E2/E3 final RL: format-valid rate</h2>
    <div id="chart-fmt3"></div>
    <div class="legend" id="leg-fmt3"></div>
  </div>
</div>

<div class="tip" id="tip"></div>

<script>
const DATA = __DATA__;

const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const COLORS = () => [css('--c1'), css('--c2'), css('--c3'), css('--c4'), css('--c5')];
const PT_COLORS = { 'P1W1': 0, 'E2W1': 1, 'E3-P2': 2, 'E1-U-P2': 4 };
const RL_COLORS = { 'E1-U': 3, 'E1-D': 4, 'E1-U leg2': 2, 'P1W1-band': 1, 'P1W1-band lr1e-4': 0, 'E1-D leg2': 0 };
const RL3_COLORS = { 'E2-U': 0, 'E2-D': 1, 'E3-U': 2, 'E3-D': 4 };
const SIDE_COLORS = { 'B2H-U': 0, 'P1W1-band': 3, 'P1W1-band lr1e-4': 2 };
const W2_COLORS = { 'interleaved (eval traces, k=16)': 0, 'pretrain-then-RL (10B, one run)': 1, 'pretrain-then-RL (10B, two runs)': 2, 'interleaved (rollout traces)': 3 };
const W2CTL_COLORS = { 'pretrain-then-RL (one run), 3,000 updates': 1, 'pretrain-then-RL (two runs), 3,000 updates': 2 };

function statusTable() {
  const t = document.getElementById('status-table');
  t.innerHTML = '<tr><th>Stage</th><th>What it is</th><th>State</th><th>Note</th></tr>' +
    DATA.status.map(r =>
      `<tr><td>${r.name}</td><td>${r.desc}</td>` +
      `<td><span class="pill ${r.state}">${r.state}</span></td><td>${r.note}</td></tr>`
    ).join('');
}

function evalTable() {
  const t = document.getElementById('eval-table');
  const pct = v => v == null ? '—' : (100 * v).toFixed(1) + '%';
  const colors = COLORS();
  const rows = DATA.eval.map((r) => {
    const bar = r.p16 == null ? '' :
      `<div class="barwrap"><div class="bar" style="width:${100 * r.p16}%;background:${colors[0]};opacity:0.35"></div>` +
      `<div class="bar" style="width:${100 * r.p1}%;background:${colors[0]}"></div></div>`;
    return `<tr><td><b>${r.arm}</b></td><td>${r.desc}</td>` +
      `<td class="num">${pct(r.p1)}</td><td class="num">${pct(r.p8)}</td>` +
      `<td class="num">${pct(r.p16)}</td><td class="num">${pct(r.fmt)}</td>` +
      `<td style="min-width:140px">${bar}</td></tr>`;
  });
  t.innerHTML = '<tr><th>Arm</th><th>Recipe</th><th class="num">pass@1</th>' +
    '<th class="num">pass@8</th><th class="num">pass@16</th><th class="num">format</th>' +
    '<th>pass@1 (solid) / pass@16 (light)</th></tr>' + rows.join('');
}

function lineChart(elId, legId, series, xKey, yKey, opts) {
  const el = document.getElementById(elId);
  const W = 520, H = 260, m = { l: 46, r: 12, t: 10, b: 30 };
  series = Object.fromEntries(Object.entries(series).map(
    ([k, v]) => [k, v.filter(p => p[yKey] != null)]));
  const names = Object.keys(series).filter(k => series[k].length);
  const colors = COLORS();
  let xs = [], ys = [];
  names.forEach(k => series[k].forEach(p => { xs.push(p[xKey]); ys.push(p[yKey]); }));
  if (!xs.length) { el.innerHTML = '<div class="note">no data yet</div>'; return; }
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  let ymin = opts.ymin ?? Math.min(...ys), ymax = opts.ymax ?? Math.max(...ys);
  if (ymax === ymin) ymax = ymin + 1;
  const pad = (ymax - ymin) * 0.06; ymin -= (opts.ymin == null ? pad : 0); ymax += pad;
  const X = v => m.l + (v - xmin) / (xmax - xmin || 1) * (W - m.l - m.r);
  const Y = v => H - m.b - (v - ymin) / (ymax - ymin) * (H - m.t - m.b);
  let g = '';
  const yticks = 4;
  for (let i = 0; i <= yticks; i++) {
    const v = ymin + (ymax - ymin) * i / yticks, y = Y(v);
    g += `<line x1="${m.l}" y1="${y}" x2="${W - m.r}" y2="${y}" stroke="${css('--grid')}" stroke-width="1"/>`;
    g += `<text x="${m.l - 6}" y="${y + 4}" text-anchor="end" font-size="10.5" fill="${css('--muted')}">${opts.yfmt(v)}</text>`;
  }
  const xticks = 5;
  for (let i = 0; i <= xticks; i++) {
    const v = xmin + (xmax - xmin) * i / xticks, x = X(v);
    g += `<text x="${x}" y="${H - m.b + 16}" text-anchor="middle" font-size="10.5" fill="${css('--muted')}">${Math.round(v)}</text>`;
  }
  g += `<text x="${(m.l + W - m.r) / 2}" y="${H - 4}" text-anchor="middle" font-size="11" fill="${css('--ink2')}">${opts.xlabel}</text>`;
  names.forEach((k) => {
    const c = colors[opts.colorMap[k]];
    const pts = series[k];
    const d = pts.map((p, i) => (i ? 'L' : 'M') + X(p[xKey]).toFixed(1) + ',' + Y(p[yKey]).toFixed(1)).join('');
    g += `<path d="${d}" fill="none" stroke="${c}" stroke-width="2" stroke-linejoin="round"/>`;
    const last = pts[pts.length - 1];
    g += `<circle cx="${X(last[xKey])}" cy="${Y(last[yKey])}" r="3" fill="${c}"/>`;
  });
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" data-chart="${elId}">${g}</svg>`;
  const leg = document.getElementById(legId);
  leg.innerHTML = names.map(k =>
    `<span style="--sw:${colors[opts.colorMap[k]]}">${k}</span>`).join('');

  // nearest-point hover tooltip
  const svg = el.querySelector('svg');
  const tip = document.getElementById('tip');
  svg.addEventListener('mousemove', (ev) => {
    const r = svg.getBoundingClientRect();
    const mx = (ev.clientX - r.left) / r.width * W;
    const my = (ev.clientY - r.top) / r.height * H;
    let best = null, bd = 1e9;
    names.forEach(k => series[k].forEach(p => {
      const dx = X(p[xKey]) - mx, dy = Y(p[yKey]) - my, d2 = dx * dx + dy * dy;
      if (d2 < bd) { bd = d2; best = { k, p }; }
    }));
    if (best && bd < 900) {
      tip.style.display = 'block';
      tip.style.left = (ev.clientX + 12) + 'px';
      tip.style.top = (ev.clientY - 10) + 'px';
      tip.textContent = `${best.k} · ${opts.xlabel} ${best.p[xKey]} · ${opts.yfmt(best.p[yKey])}`;
    } else tip.style.display = 'none';
  });
  svg.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
}

function evlossTable() {
  const t = document.getElementById('evloss-table');
  t.innerHTML = '<tr><th>Endpoint</th><th class="num">PT held-out</th>' +
    '<th class="num">SFT P1-half</th><th class="num">SFT P2-half</th><th class="num">Uniform</th></tr>' +
    DATA.evloss.map(r =>
      `<tr><td><b>${r.tag}</b></td><td class="num">${r.pt.toFixed(3)}</td>` +
      `<td class="num">${r.sft_p1.toFixed(3)}</td>` +
      `<td class="num">${r.sft_p2.toFixed(3)}${r.p2_unseen ? ' (unseen)' : ''}</td>` +
      `<td class="num">${r.uniform.toFixed(3)}</td></tr>`
    ).join('');
}

function render() {
  statusTable();
  evalTable();
  evlossTable();
  lineChart('chart-uni', 'leg-uni', DATA.loss, 'step', 'uni',
    { xlabel: 'optimizer step', yfmt: v => v.toFixed(2), colorMap: PT_COLORS });
  lineChart('chart-pt', 'leg-pt', DATA.loss, 'step', 'pt',
    { xlabel: 'optimizer step', yfmt: v => v.toFixed(2), colorMap: PT_COLORS });
  lineChart('chart-sft', 'leg-sft', DATA.loss, 'step', 'sft',
    { xlabel: 'optimizer step', yfmt: v => v.toFixed(2), colorMap: PT_COLORS });
  lineChart('chart-p1c', 'leg-p1c', DATA.passcurve, 'step', 'p1',
    { xlabel: 'RL step', yfmt: v => (100 * v).toFixed(1) + '%', colorMap: RL_COLORS });
  lineChart('chart-p16c', 'leg-p16c', DATA.passcurve, 'step', 'p16',
    { xlabel: 'RL step', yfmt: v => (100 * v).toFixed(1) + '%', colorMap: RL_COLORS });
  lineChart('chart-rww2', 'leg-rww2', DATA.rlwave2, 'rollout', 'reward',
    { xlabel: 'rollout update', yfmt: v => v.toFixed(3), ymin: 0, colorMap: W2_COLORS });
  lineChart('chart-p1w2', 'leg-p1w2', DATA.passcurve_w2, 'step', 'p1',
    { xlabel: 'RL step', yfmt: v => (100 * v).toFixed(1) + '%', colorMap: W2_COLORS });
  lineChart('chart-p16w2', 'leg-p16w2', DATA.passcurve_w2, 'step', 'p16',
    { xlabel: 'RL step', yfmt: v => (100 * v).toFixed(1) + '%', colorMap: W2_COLORS });
  lineChart('chart-rww2c', 'leg-rww2c', DATA.rlw2ctl, 'rollout', 'reward',
    { xlabel: 'rollout update', yfmt: v => v.toFixed(3), ymin: 0, colorMap: W2CTL_COLORS });
  lineChart('chart-p1side', 'leg-p1side', DATA.passcurve_side, 'step', 'p1',
    { xlabel: 'RL step', yfmt: v => (100 * v).toFixed(1) + '%', colorMap: SIDE_COLORS });
  lineChart('chart-p16side', 'leg-p16side', DATA.passcurve_side, 'step', 'p16',
    { xlabel: 'RL step', yfmt: v => (100 * v).toFixed(1) + '%', colorMap: SIDE_COLORS });
  lineChart('chart-rwside', 'leg-rwside', DATA.rlside, 'rollout', 'reward',
    { xlabel: 'rollout update', yfmt: v => v.toFixed(3), ymin: 0, colorMap: SIDE_COLORS });
  lineChart('chart-p1c3', 'leg-p1c3', DATA.passcurve3000, 'step', 'p1',
    { xlabel: 'RL step', yfmt: v => (100 * v).toFixed(1) + '%', colorMap: RL3_COLORS });
  lineChart('chart-p16c3', 'leg-p16c3', DATA.passcurve3000, 'step', 'p16',
    { xlabel: 'RL step', yfmt: v => (100 * v).toFixed(1) + '%', colorMap: RL3_COLORS });
  lineChart('chart-rw', 'leg-rw', DATA.rl, 'rollout', 'reward',
    { xlabel: 'rollout update', yfmt: v => v.toFixed(3), ymin: 0, colorMap: RL_COLORS });
  lineChart('chart-fmt', 'leg-fmt', DATA.rl, 'rollout', 'fmt',
    { xlabel: 'rollout update', yfmt: v => (100 * v).toFixed(0) + '%', ymin: 0, ymax: 1, colorMap: RL_COLORS });
  lineChart('chart-rw3', 'leg-rw3', DATA.rl3000, 'rollout', 'reward',
    { xlabel: 'rollout update', yfmt: v => v.toFixed(3), ymin: 0, colorMap: RL3_COLORS });
  lineChart('chart-fmt3', 'leg-fmt3', DATA.rl3000, 'rollout', 'fmt',
    { xlabel: 'rollout update', yfmt: v => (100 * v).toFixed(0) + '%', ymin: 0, ymax: 1, colorMap: RL3_COLORS });
}
render();
if (window.matchMedia) window.matchMedia('(prefers-color-scheme: dark)')
  .addEventListener('change', render);
</script>
</body>
</html>
"""

html = HTML.replace("__DATA__", json.dumps(data)).replace("__GENERATED__", data["generated"])
OUT.write_text(html)
print(f"wrote {OUT} ({len(html) // 1024} KB)")
print("loss points:", {k: len(v) for k, v in loss_series.items()})
print("rl points:", {k: len(v) for k, v in rl_series.items()})
