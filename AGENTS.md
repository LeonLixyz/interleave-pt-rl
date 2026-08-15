# Production training contract

This repository contains the canonical interleaved pretraining, SFT, RL, and
evaluation pipelines. Treat numerical precision, data identity, checkpoint
identity, and launch provenance as correctness requirements. Throughput is
secondary to those requirements.

## Mixed precision

All production PT, SFT, mixed PT+SFT, and RL training must use the following
accuracy-first BF16 mixed-precision contract unless an experiment explicitly
studies a different precision:

- Optimizer-facing main parameters are FP32.
- Adam first and second moments are FP32.
- Forward and backward computation uses BF16.
- Gradient accumulation and distributed reduction use FP32.
- Resumable training checkpoints preserve actual FP32 parameter and optimizer
  tensors. A config field that says `float32` is not proof of tensor dtype.
- Inference and rollout may cast the same canonical FP32 checkpoint to BF16 in
  memory. A separately persisted BF16 model is optional and must never replace
  the FP32 resumable checkpoint.
- Loading a BF16 checkpoint and continuing to optimize BF16 tensors is forbidden
  for production training. Explicitly upcast before constructing FSDP and the
  optimizer, or fail closed.

Every trainer must assert and log the real parameter dtype before wrapping, the
main/sharded parameter dtype after wrapping, gradient reduction dtype, optimizer
moment dtypes after the first update, and checkpoint tensor dtypes after saving.
Do not infer these from autocast settings, configuration, filenames, or file
sizes.

## Required infrastructure gates

Do not launch a production experiment after changing a trainer, tokenizer,
checkpoint, rollout, reward, or distributed-training path until all applicable
gates pass:

1. Unit tests for the changed contract and its failure cases.
2. A Modal GPU canary using the exact production image and entry point.
3. At least one real optimizer update with persisted inspection showing FP32
   main parameters and FP32 Adam moments.
4. Save, reload, and resume verification with the next update producing the same
   precision contract.
   For staged SFT, test this in a fresh process from each supported PT parent
   transition (81-to-85 and 85-to-85), and compare the resumed two-update model
   bit-for-bit with an uninterrupted reference run.
5. A clean HF export whose tensors are FP32, followed by inference that loads
   that same export and casts it to BF16 in memory.
6. For RL, one complete rollout and policy update verifying exactly one BOS,
   correct prompt/response/environment masking, fresh rollout weight versions,
   finite metrics, and reward agreement with the offline scorer.

A successful process exit is not sufficient. Inspect the saved tensor metadata
and record the evidence in the run provenance.

Resumable checkpoints must be published as immutable step directories. Write
an authenticated completion marker only after every model, optimizer,
scheduler, RNG, cursor, and metadata artifact is durable, then atomically move
the latest pointer to that completed step. Never resume from a directory that
has no valid completion marker, and never update a committed step in place.

## Checkpoints and experiment identity

- Training roots, run names, W&B IDs, filtered datasets, and provenance records
  are immutable once used.
- Any change to parameter precision, optimizer precision, tokenizer, dataset,
  loss, batch geometry, context length, or code manifest requires a new
  experiment version and fresh output roots.
- Never resume a run whose stored precision or provenance contract differs from
  the requested run. Fail with a clear error instead.
- Preserve invalid or superseded runs for diagnosis, but label them explicitly
  and exclude them from canonical comparisons.
- Record the exact launch command, code manifest hash, container image, source
  checkpoint manifest, tokenizer files and IDs, dataset path and SHA-256,
  precision contract, seed, optimizer settings, loss settings, batch geometry,
  token budget, and target updates.
- A stage initialized from another checkpoint must persist and authenticate the
  complete parent export identity, not only its path or tensor shapes. Resume
  and final validation must reject a different same-shaped parent.

## Chess sequence semantics

- The 81-token and 85-token scratch models must use one canonical seeded
  85-row initialization.  With the same seed, every non-vocabulary parameter
  and embedding rows 0:81 must be bitwise identical; rows 81:85 must also be
  deterministic.  Do not initialize the two vocabulary shapes from separate
  shape-dependent random streams.
- Every packed PT sequence starts with one explicitly prepended `<bos>` context
  token while preserving the exact historical supervised target tokens.
  Source-document boundary tokens may still occur inside packed PT targets.
- SFT, evaluation, filtering, and RL prompts start with exactly one `<bos>`.
  The tokenizer does not add it implicitly.
- Prompts are not supervised. Generated reasoning and move tokens are
  supervised. Environment replies are masked from the loss.
- Separate SFT rows remain one row per right-padded sequence. Mixed training must
  not pack an SFT row together with PT text.
- The final tokenizer used by chess SFT, evaluation, filtering, and RL must have
  the expected complete token-to-ID mapping and be authenticated by file
  hashes. Checking only special-token IDs is insufficient.
- A controlled vocabulary comparison must initialize every shared parameter
  bit-for-bit identically. Construct from one canonical initialization and
  resize or slice vocabulary rows; do not rely on the same seed across
  shape-dependent model constructors.
- Offline filtering and online reward must use the same move extraction and
  scoring semantics. Test black and white castling explicitly.
- The model's native context, the policy-training limit, and SGLang's actual
  runtime `context_length` must agree. For the context-2,048 experiments they
  are all exactly 2,048; verify the live server setting rather than only the
  launch command.

## RL update semantics

- The canonical chess geometry is 256 prompts times 8 samples: 2,048
  trajectories and one optimizer update per rollout batch.
- Token-mean loss means one global supervised-token denominator for the complete
  optimizer update. Do not compute independent microbatch means and then sum
  them.
- Microbatching or packing changes must be tested for gradient equivalence and
  recorded in provenance.
- Assert that rollout weight versions advance after every optimizer update and
  that no active FunctionCall is duplicated during recovery.
- A cross-stage Adam continuation is a separate experiment, never an implicit
  default. Authenticate the complete parent optimizer checkpoint, require the
  RL model weights to be byte-identical to the optimizer source, map every
  parent parameter ID to one exact model parameter name, and reject partial or
  shape-only loads. Carry FP32 `exp_avg`, `exp_avg_sq`, and the per-parameter
  Adam step while starting the RL rollout/update cursor at zero. Unless the
  experiment explicitly changes them, retain the destination RL learning rate,
  betas, epsilon, and weight decay so optimizer-state continuation is the only
  changed variable. Gate the first update and a fresh-process resume, and prove
  that every Adam step equals `parent_step + RL_updates`.

## Modal launches

- Launch long jobs with `modal run --detach` or an equivalent deployed-function
  spawn whose lifetime is independent of the local process.
- Production runs require an immutable, version-scoped atomic launch claim
  and a Volume-backed durable launch anchor before allocating a GPU. Scope the
  durable anchor to the immutable output identity, not to a source revision;
  keep the source hash inside the anchor so a redeploy cannot hide an older
  launch. The worker must reload and authenticate the anchor and claim token,
  then bind them to exactly one Modal FunctionCall. Refresh the validated Dict
  lease throughout training, but never infer that a stale heartbeat permits
  claim takeover. Dict expiry or loss must fail closed when a durable anchor or
  output root exists.
- Long-running production functions must use a Modal timeout of at most 24
  hours and publish resumable checkpoints frequently enough to survive it.
- Before production claims are accepted, a source-scoped infrastructure gate
  must perform a real W&B write in the exact entity and project, finish it,
  read the exact run back, and authenticate its persisted marker. Keep this
  gate out of production run groups.
- Preserve the raw launch token in one mode-0600 local recovery record created
  with exclusive final-path creation and file plus directory `fsync`. Never
  print or transmit the raw token except as the authenticated dispatcher and
  worker argument.
- A retry after any launch attempt requires the original raw token, unchanged
  launch identity, and a new immutable attempt generation acquired by atomic
  compare-and-set. Recovery must poll the exact retained FunctionCall result
  with `FunctionCall.get(timeout=0)`; Modal call graphs are best-effort and must
  never authorize recovery. Running, successful, unknown, ambiguous, or expired
  results fail closed. Re-authenticate the committed Volume checkpoint only
  after the prior worker is authoritatively terminal, then bind that evidence
  to the next generation. A worker must authenticate that its generation is
  current, so a delayed worker from an older generation cannot begin training.
- Worker binding and recovery closure must contend on one immutable atomic
  decision for each generation. If recovery closes first, every delayed worker
  is rejected before training. If a worker binds first, recovery must prove
  that exact worker FunctionCall terminal and unsuccessful before advancing.
- Claiming, attempt generation, spawning, and immediate FunctionCall binding
  belong in one deployed CPU dispatcher. Concurrent recoveries must yield at
  most one new generation and one GPU spawn. Disable automatic dispatcher
  retries; same-dispatcher replay may return an already-bound worker but must
  not report success for an unbound attempt. Never steal based on age.
- Use one deterministic W&B run ID across retries and authenticated resumes.
  A training function may return success only after it commits, reloads, and
  authenticates the checkpoint at the exact requested target, then publishes
  and reads back an immutable Volume-backed completion record.
- Modal Volume commits snapshot the entire mount. Coordinate trainer staging
  and launcher commits with an explicit shared lock or equivalent handshake;
  do not commit while checkpoint, snapshot, metric, or HF-export staging is in
  progress.
- Before launching, resolve the exact checkpoint, output root, dataset hash,
  W&B identity, and active calls. Refuse collisions.
- Immediately verify the FunctionCall is live and W&B is reporting.
- A retry or resume must preserve byte-for-byte experiment semantics and pass
  the stored provenance guard.

## Change discipline

- Fix infrastructure defects before interpreting algorithmic results.
- Prefer fail-closed assertions over warnings for precision, provenance, data,
  tokenizer, and checkpoint invariants.
- Keep user-facing experiment names literal and descriptive. Do not invent
  shorthand labels.
- Never present a run as canonical until every required gate above has passed.
