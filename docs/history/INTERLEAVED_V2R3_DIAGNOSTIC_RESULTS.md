# Interleaved v2r3 diagnostic results

Authenticated at `2026-07-30T15:57:17.299166+00:00`.

## Immutable result

- Audit call:
  `fc-01KYSVTZ9AE62710QPYS5CG2XJ`
- Storage path:
  `/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/v2r3_diagnostic_20260730/seed42_report.json`
- Logical dashboard path:
  `/artifacts/interleave_50m/v2r3_diagnostic_20260730/seed42_report.json`
- Schema:
  `interleaved-v2r3-diagnostic-report-v1`
- Records: 12 ordered, unique snapshot/rollout records
- Canonical report self-hash:
  `d803e26631c15623892c2a9d029fa0b37b689cde72982dbefe2a5baebde6d16c`
- File SHA-256:
  `fe3b5836a65cebf565840f5b790b1c6aee833fc7616cd9c32e740f6fe1abfffc`
- File size: 382,465 bytes

The embedded self-hash was independently recomputed from the canonical report
core and matched. Every record binds one authenticated training call,
resume/HF snapshot identities, one successful rollout call, and all four
rollout artifact hashes. Each rollout contains exactly 256 prompt groups × 8
siblings = 2,048 rows with the frozen seed-42 prompt-set identity.

This report remains diagnostic-only:

- `production_authorized=false`
- `p1_authorized=false`
- `exp2_authorized=false`

## Frozen aggregate results

PT and SFT CE are token-weighted, pre-update training-stream interval
measurements. They are not held-out or endpoint-checkpoint evaluations.
`Positive rate` is the diagnostic binary-reward rate, not official Pass@1.

| SFT weight | Step | PT CE | SFT CE | Protocol rows | Protocol rate | Positives | Solve given protocol | Positive rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 190.189290837 | 1,000 | 1.7006 | 1.5908 | 179 | 8.740% | 0 | 0.000% | 0.000% |
| 190.189290837 | 2,000 | 0.9278 | 0.8101 | 45 | 2.197% | 0 | 0.000% | 0.000% |
| 190.189290837 | 4,000 | 0.8006 | 0.7094 | 742 | 36.230% | 9 | 1.213% | 0.439% |
| 190.189290837 | 6,000 | 0.7202 | 0.6479 | 431 | 21.045% | 15 | 3.480% | 0.732% |
| 190.189290837 | 8,000 | 0.6791 | 0.6062 | 266 | 12.988% | 9 | 3.383% | 0.439% |
| 190.189290837 | 9,920 | 0.6686 | 0.5972 | 219 | 10.693% | 8 | 3.653% | 0.391% |
| 256 | 1,000 | 1.7737 | 1.6288 | 33 | 1.611% | 0 | 0.000% | 0.000% |
| 256 | 2,000 | 0.9805 | 0.8251 | 67 | 3.271% | 2 | 2.985% | 0.098% |
| 384 | 1,000 | 1.9048 | 1.6986 | 726 | 35.449% | 1 | 0.138% | 0.049% |
| 384 | 2,000 | 1.0489 | 0.8512 | 363 | 17.725% | 3 | 0.826% | 0.146% |
| 768 | 1,000 | 2.1637 | 1.8435 | 27 | 1.318% | 0 | 0.000% | 0.000% |
| 768 | 2,000 | 1.2000 | 0.8899 | 145 | 7.080% | 0 | 0.000% | 0.000% |

At step 2,000, relative to weight 190.189290837:

| Weight | PT CE increase | SFT CE increase |
| ---: | ---: | ---: |
| 256 | 5.68% | 1.84% |
| 384 | 13.06% | 5.06% |
| 768 | 29.34% | 9.85% |

Thus the larger weights had higher observed CE for both training-stream
objectives at the matched early steps. Their 0–3 positive outcomes were too
sparse to establish a compensating behavioral benefit.

## Paired read-only analysis

This secondary analysis is not part of the immutable report and does not
change it. All 12 authenticated JSONL artifacts were downloaded by exact
known path, and every file SHA matched its report entry. Across snapshots,
all 2,048 row identities matched on:

- sample and group index;
- sibling index and seed;
- puzzle ID;
- exact input string.

This permits exploratory paired comparisons on the fixed prompt/sibling set.
The table uses post-hoc, unadjusted exact two-sided McNemar tests; the displayed
p-values have no multiplicity correction and are not gate thresholds.
Prompt-group comparisons use whether any of the eight siblings solved the
prompt, which avoids treating all siblings as independent.

| Weight-190 comparison | Positive samples | Lost / gained samples | Exact p | Solved prompt groups | Lost / gained groups | Exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 → 2,000 | 0 → 0 | 0 / 0 | 1.000 | 0 → 0 | 0 / 0 | 1.000 |
| 2,000 → 4,000 | 0 → 9 | 0 / 9 | 0.0039 | 0 → 8 | 0 / 8 | 0.0078 |
| 4,000 → 6,000 | 9 → 15 | 8 / 14 | 0.286 | 8 → 12 | 5 / 9 | 0.424 |
| 6,000 → 8,000 | 15 → 9 | 15 / 9 | 0.307 | 12 → 9 | 6 / 3 | 0.508 |
| 8,000 → 9,920 | 9 → 8 | 6 / 5 | 1.000 | 9 → 8 | 5 / 4 | 1.000 |
| 6,000 → 9,920 | 15 → 8 | 15 / 8 | 0.210 | 12 → 8 | 9 / 5 | 0.424 |

The fixed-prompt result is consistent with the first detected rise between
steps 2,000 and 4,000 on this single diagnostic set. It does not establish
that step 6,000 is better than the full step-9,920 checkpoint. In particular,
the 15 step-6,000 positive sibling rows and 8 step-9,920 positive sibling rows
do not overlap. Only three solved prompt groups overlap. The apparent peak is
therefore noisy and identity-unstable.

At step 2,000, every pairwise exact test among weights
190.189290837/256/384/768 has `p >= 0.25` for positive samples. The behavior
probe does not select a weight at that step. Protocol-format rates differ
strongly but non-monotonically; protocol compliance alone is not task
success.

For descriptive reference, row-level Wilson 95% intervals for the observed
positive rates are:

- 8/2,048: 0.198%–0.769%;
- 9/2,048: 0.231%–0.833%;
- 15/2,048: 0.444%–1.205%.

These intervals overlap substantially. They treat sibling rows as independent
and therefore do not account for within-prompt clustering.

## Interpretation

1. Weight 190.189290837 has the lowest observed matched early training-stream
   PT and SFT CE among these single-seed step-1,000/2,000 trajectories.
2. Useful behavior appears only after more than 2,000 steps in the long
   weight-190 trajectory.
3. Step 6,000 has the highest point estimate for diagnostic positive rate,
   but the paired evidence does not distinguish it from steps 8,000 or 9,920.
4. Continued training through step 9,920 keeps improving PT/SFT interval CE.
   The final checkpoint has the highest conditional solve rate among
   protocol-valid rows, but fewer protocol-valid outputs, so its total
   positive-rate point estimate is below step 6,000.
5. No final-pretraining-performance conclusion is possible without a frozen
   held-out or endpoint benchmark.
6. No production, P1, Exp2, or RL launch is authorized by this diagnostic.

## Proposed follow-up gate (not launched)

A separately frozen production decision should:

1. evaluate the existing weight-190 checkpoints at steps 6,000, 8,000, and
   9,920 on two new prompt-disjoint batches;
2. use new frozen prompt fingerprints/seeds and never reuse the seed-42
   diagnostic batch for selection;
3. predeclare prompt-group-aware paired statistics and multiplicity handling;
4. add frozen held-out PT/SFT or endpoint pretraining benchmarks;
5. predeclare the joint behavior/pretraining selection rule before reading
   either new batch.

For an effect the size of the observed step-6,000 versus step-9,920 point
estimate, a naive independent-row calculation needs about 7,500 rows per
checkpoint for 80% power. Multiplying by the largest observed row-level
prompt-cluster design effect gives a heuristic of about 10,200 rows, but that
is not sufficient for a prompt-level primary endpoint. Under proportional
replication of the observed prompt-group discordance, an exact paired
McNemar design needs about 1,851 prompt groups, or 14,808 rows per checkpoint,
for approximately 80% power.

The safer launch shape is therefore 2,048 prompt groups × 8 siblings = 16,384
rows per checkpoint, split evenly across two disjoint 1,024-prompt batches.
The exact size, endpoint, multiplicity rule, and selection rule must still be
frozen before launch.
