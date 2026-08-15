# Results

Measurement outputs and research notes. Nothing here is needed to run the code —
`docs/` covers that.

## Caches

Everything behind the two published pages, so they can be rebuilt with no GPU work:

| File | Contents |
|---|---|
| `rl_pass_curve_full.json` | every merged stride-8 pass@k curve point |
| `rl_curves.jsonl`, `wave2_curves*.jsonl`, `rl3000r_curves.jsonl` | per-update reward and format-rate series |
| `ptloss_*.out` | held-out pretraining loss and SFT loss measurements |
| `rl_eval_ledger.txt`, `rl_ptloss_ledger.txt` | which (run, step) points have been evaluated |

Rebuild the pages with `tools/build_f_section.py` → `tools/splice_panels.py` →
`modal deploy dashboard/deploy_interleave_results.py`, and `tools/build_dashboard.py`
→ `modal deploy dashboard/deploy_interleave_pt_rl.py`.

Live pages:
- https://modal-labs-leon-dev--interleave-results-web.modal.run
- https://modal-labs-leon-dev--interleave-pt-rl-web.modal.run

## Notes

| File | Contents |
|---|---|
| `notes/findings.md` | what the research programme found, model lineage, vocabulary, caveats |
| `notes/rl-throughput-and-parity.md` | why Miles RL is faster than the old verl setup, and where the two configs still differ |
| `notes/lessons.md` | incidents and failure modes worth knowing before trusting a number |
