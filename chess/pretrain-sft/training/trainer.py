import pathlib, shutil
import os, sys, argparse, pathlib
import time
from tqdm import tqdm

repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from accelerate import Accelerator
from .data_utils import create_dataloader
from model.gpt2_model import GPT, GPTConfig
from llm_tokens.chess.tokenizer_factory import init_tokenizer
from .optim_sched import build_optimizer, build_scheduler
import torch
from evaluation.example_evaluator import HumanGamesEvaluator, PuzzlesEvaluator, RandomGamesEvaluator
from evaluation.metrics import calculate_cp_gap

class Trainer:
    def __init__(self, cfg, run_config_path=None):
        self.cfg = cfg
        self.run_cfg_path = run_config_path
        backend = cfg.logging.get("backend", "wandb")
        # Enable mixed precision training for better GPU utilization
        mixed_precision = cfg.training.get("mixed_precision", "no")  # bf16, fp16, or no
        gradient_accumulation_steps = cfg.training.get("gradient_accumulation_steps", 1)
        self.acc = Accelerator(
            log_with=("wandb" if backend == "wandb" else backend),
            mixed_precision=mixed_precision if mixed_precision != "no" else None,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        self._init_all()

    # ---------------- core builders ----------------
    def _init_all(self):
        cfg = self.cfg
        self.mcfg, self.tcfg, self.dcfg, self.tokcfg = cfg.model, cfg.training, cfg.data, cfg.tokenizer

        # ----- tokenizer -----
        self.tok = init_tokenizer(
            name=self.tokcfg.name,
            config=self.tokcfg
        )

        # ----- dataloaders -----
        all_files = self._expand_txt_files(self.dcfg)

        if "eval_txt_files" in self.dcfg and self.dcfg.eval_txt_files:
            # explicit eval files (expand them too)
            self.eval_files = self._expand_txt_files({"txt_files": self.dcfg.eval_txt_files})
            self.train_files = [p for p in all_files if p not in set(self.eval_files)]
        else:
            k = int(self.dcfg.get("eval_holdout", 1))
            k = max(0, min(k, len(all_files) - 1))
            self.train_files = all_files[:-k] if k > 0 else all_files
            self.eval_files = all_files[-k:] if k > 0 else []

        # Deterministic seed for shard shuffling (reproducible across resume)
        self._data_seed = int(self.tcfg.get("seed", 42))

        # Optimize data loading for better GPU utilization
        num_workers = int(self.tcfg.get("num_workers", 4))  # Default to 4 workers
        prefetch_factor = self.tcfg.get("prefetch_factor", 2 if num_workers > 0 else None)
        persistent_workers = self.tcfg.get("persistent_workers", True if num_workers > 0 else False)

        self.train_loader = create_dataloader(
                txt_files=self.train_files,
                tokenizer=self.tok,
                batch_size=self.tcfg.batch_size,
                seq_len=self.mcfg.block_size,
                num_workers=num_workers,
                shuffle=False,
                cache_size=self.tcfg.batch_size * self.mcfg.block_size * 64,
                dataset_shuffle=True,
                num_shards=self.dcfg.get("num_shards", None),
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
                seed=self._data_seed,
            )
        
        # Store these for later use when recreating dataloader
        self.dataloader_kwargs = {
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
            "persistent_workers": persistent_workers,
        }

        self.eval_loader = None
        if self.eval_files:
            self.eval_loader = create_dataloader(
            txt_files=self.eval_files,
            tokenizer=self.tok,
            batch_size=self.tcfg.get("eval_batch_size", self.tcfg.batch_size),
            seq_len=self.mcfg.block_size,
            num_workers=0,
            shuffle=False,
        )

        # vocab size (tolerate both interfaces)
        if hasattr(self.tok, "get_vocab_size"):
            self.vocab_size = int(self.tok.get_vocab_size())
            print("Vocab size: ", self.vocab_size)
        else:
            self.vocab_size = int(len(self.tok.get_vocab()))

        # ----- model -----
        self.model = GPT(GPTConfig(
            vocab_size=self.vocab_size,
            block_size=self.mcfg.block_size,
            n_layer=self.mcfg.n_layer,
            n_head=self.mcfg.n_head,
            n_embed=self.mcfg.n_embed,
            dropout=self.mcfg.dropout,
            mlp_type=self.mcfg.mlp_type,
        ))
        
        # Load pretrained weights if specified (before accelerator.prepare)
        if "pretrained_weights" in self.tcfg and self.tcfg.pretrained_weights:
            self._load_pretrained_weights(self.tcfg.pretrained_weights)
        
        # Optional: Compile model for additional speedup (PyTorch 2.0+)
        if self.tcfg.get("compile_model", False):
            compile_mode = self.tcfg.get("compile_mode", "reduce-overhead")
            self.acc.print(f"[optimization] Compiling model with mode={compile_mode}")
            try:
                import torch
                if hasattr(torch, 'compile'):
                    self.model = torch.compile(self.model, mode=compile_mode)
                    self.acc.print("[optimization] Model compiled successfully (first iteration will be slower)")
                else:
                    self.acc.print("[warn] torch.compile not available, skipping compilation")
            except Exception as e:
                self.acc.print(f"[warn] Model compilation failed: {e}")
        
        print("="*50)
        print("Model: ", self.model)
        print("="*50)

        # ----- optimizer & scheduler -----
        self.optimizer = build_optimizer(self.model, {
            "name": self.tcfg.optimizer.name,
            "lr": self.tcfg.optimizer.lr,
            "weight_decay": self.tcfg.optimizer.get("weight_decay", 0.1),
            "betas": tuple(self.tcfg.optimizer.get("betas", (0.9, 0.95))),
            "exclude_bias_and_norm": True,
        })

        self.epochs = self.tcfg.epochs
        steps_per_epoch = len(self.train_loader)
        self.scheduler = build_scheduler(
            self.optimizer,
            {
                "name": self.tcfg.scheduler.name,
                "eta_min": self.tcfg.scheduler.get("eta_min", 0.0),
                "warmup_steps": self.tcfg.scheduler.warmup_steps
            },
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
            grad_accum_steps=self.acc.gradient_accumulation_steps,
        )

        # ----- accelerator prep -----
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.scheduler,
            self.eval_loader,
        ) = self.acc.prepare(self.model, self.optimizer, self.train_loader, self.scheduler, self.eval_loader)

        self.acc.print(f"[accelerate] Using device: {self.acc.device}")

        # ----- dirs & tracking -----
        run_name = self._auto_run_name(cfg)
        self.save_dir = pathlib.Path(self.tcfg.get("save_dir", "checkpoints/gpt"))
        self.save_dir = self.save_dir / run_name
        self.save_interval = self.tcfg.get("save_interval", 1000)
        self.log_interval = self.tcfg.get("log_interval", 50)
        self.resume_path = self.tcfg.get("resume", None)
        self.smoke_test = bool(self.tcfg.get("smoke_test", False))
        
        # Training state for resuming
        self.current_step = 0
        self.current_epoch = 0

        if self.acc.is_main_process:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # init trackers on ALL ranks (Accelerate will manage rank-0 setup)
        project_name = self.cfg.logging.get("project", "llm-training")
        self.acc.print(f"Initing trackers: project={project_name} run={run_name}")
        self.acc.init_trackers(
            project_name,                      
            init_kwargs=self._tracker_init_kwargs(run_name, self._tracker_cfg())
        )
        if self.acc.is_main_process:
            import wandb
            if wandb.run is not None:
                wandb.run.name = run_name
                self.acc.print(f"[wandb] effective name: {wandb.run.name} (id={wandb.run.id})")

        if self.acc.is_main_process:
            self._snapshot_config()

        # Auto-resume: find latest checkpoint if enabled and no explicit resume path
        auto_resume = bool(self.tcfg.get("auto_resume", False))
        if not self.resume_path and auto_resume:
            self.resume_path = self._find_latest_checkpoint()
            if self.resume_path:
                self.acc.print(f"[auto_resume] Found latest checkpoint: {self.resume_path}")

        if self.resume_path:
            self.acc.load_state(self.resume_path)
            self._load_training_state(self.resume_path)
            self.acc.print(f"[train] resumed from {self.resume_path} at step {self.current_step}, epoch {self.current_epoch}")

        # MFU tracking state
        self._last_log_time = time.time()
        self._last_log_step = 0
        self._tokens_per_batch = self.tcfg.batch_size * self.mcfg.block_size
        self._flops_per_token = self._estimate_flops_per_token()
        self._gpu_peak_tflops = self.tcfg.get("gpu_peak_tflops", 312)  # A100 bf16 default

        if self.smoke_test and self.acc.is_main_process:
            xb, yb = next(iter(self.train_loader))
            self.acc.print(f"[smoke] x={tuple(xb.shape)} y={tuple(yb.shape)}")

    def _load_pretrained_weights(self, path):
        """Load only model weights from a checkpoint (not optimizer/scheduler)."""
        import torch
        from pathlib import Path
        path = Path(path)
        
        if path.is_dir():
            # Accelerate checkpoint directory structure
            model_path = path / "pytorch_model.bin"
            if not model_path.exists():
                # Try alternative names
                alt_paths = list(path.glob("*.bin")) + list(path.glob("*.pt")) + list(path.glob("*.pth"))
                if alt_paths:
                    model_path = alt_paths[0]
                else:
                    raise FileNotFoundError(f"No model weights found in {path}")
        else:
            model_path = path
        
        self.acc.print(f"[init] Loading pretrained weights from {model_path}")
        state_dict = torch.load(model_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        # Load weights with strict=False to allow partial loading
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        
        if missing:
            self.acc.print(f"[init] Missing keys: {missing}")
        if unexpected:
            self.acc.print(f"[init] Unexpected keys: {unexpected}")
        
        self.acc.print(f"[init] Successfully loaded pretrained weights")
    
    def _save_training_state(self, checkpoint_dir):
        """Save training state for proper resuming."""
        import json
        if self.acc.is_main_process:
            grad_accum = self.acc.gradient_accumulation_steps
            state_file = pathlib.Path(checkpoint_dir) / "training_state.json"
            state = {
                "step": self.current_step,
                "epoch": self.current_epoch,
                "total_batches": self.current_step * grad_accum,
                "grad_accum_steps": grad_accum,
                "data_seed": self._data_seed,
            }
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)

    def _load_training_state(self, checkpoint_dir):
        """Load training state when resuming."""
        import json
        state_file = pathlib.Path(checkpoint_dir) / "training_state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                state = json.load(f)
            self.current_step = state.get("step", 0)
            self.current_epoch = state.get("epoch", 0)
            self._data_seed = state.get("data_seed", self._data_seed)
        else:
            self.acc.print(f"[warn] No training_state.json found in {checkpoint_dir}, starting from step 0")

    def _find_latest_checkpoint(self):
        """Find the latest step checkpoint in save_dir for auto-resume."""
        import re
        if not self.save_dir.exists():
            return None
        ckpt_dirs = sorted(
            (d for d in self.save_dir.iterdir()
             if d.is_dir() and re.match(r"step\d+$", d.name)),
            key=lambda d: int(re.search(r"\d+", d.name).group()),
        )
        if not ckpt_dirs:
            return None
        latest = ckpt_dirs[-1]
        # Verify the checkpoint has required files
        if (latest / "training_state.json").exists():
            return str(latest)
        self.acc.print(f"[auto_resume] Latest checkpoint {latest} missing training_state.json, skipping")
        return None

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent max_checkpoints."""
        import re
        max_ckpts = int(self.tcfg.get("max_checkpoints", 0))
        if max_ckpts <= 0:
            return
        ckpt_dirs = sorted(
            (d for d in self.save_dir.iterdir()
             if d.is_dir() and re.match(r"step\d+$", d.name)),
            key=lambda d: int(re.search(r"\d+", d.name).group()),
        )
        while len(ckpt_dirs) > max_ckpts:
            old = ckpt_dirs.pop(0)
            self.acc.print(f"[cleanup] Removing old checkpoint: {old}")
            shutil.rmtree(old)

    # ---------------- utils ----------------
    def _auto_run_name(self, cfg):
        """Automatically build run name from model, optimizer, and data configs."""
        m = cfg.model
        t = cfg.training
        d = cfg.data
        tok = cfg.tokenizer
        warmup_steps = t.scheduler.warmup_steps
        # stem = pathlib.Path(d.txt_path).stem if "txt_path" in d else "corpus"
        stem = "_shards"
        return f"{tok.name}_include_move_numbers{tok.include_move_numbers}_data{d.num_shards}_L{m.n_layer}H{m.n_head}E{m.n_embed}_bs{t.batch_size}_lr{t.optimizer.lr}_warmup{warmup_steps}_{stem}"

    def _tracker_cfg(self):
        c = self.cfg
        return {
            "epochs": c.training.epochs,
            "batch_size": c.training.batch_size,
            "lr": c.training.optimizer.lr,
            "seq_len": c.model.block_size,
            "vocab_size": self.vocab_size,
            "tokenizer": c.tokenizer.name,
        }

    def _tracker_init_kwargs(self, run_name: str, cfg: dict):
        # accept both "wandb" and ["wandb"]
        kinds = self.acc.log_with if isinstance(self.acc.log_with, (list, tuple)) else [self.acc.log_with]
        if "wandb" not in kinds:
            return {}
        lg = self.cfg.logging
        return {
            "wandb": {
                # project is positional in init_trackers; don't duplicate here
                "entity": lg.get("entity", None),
                "name": run_name,                 
                "notes": lg.get("notes", None),
                "tags": lg.get("tags", []),
                "config": cfg,
            }
        }

    def _snapshot_config(self):
        if not self.run_cfg_path:
            return
        dst = self.save_dir / "config_snapshot" / pathlib.Path(self.run_cfg_path).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy(self.run_cfg_path, dst)
        except Exception as e:
            self.acc.print(f"[warn] config snapshot failed: {e}")

    # ---------------- train loop ----------------
    def train(self):
        self.model.train()
        steps_per_epoch = len(self.train_loader)
        resume_skip_done = False  # Flag to ensure we only skip once on resume
        gradient_accumulation_steps = self.tcfg.get("gradient_accumulation_steps", 1)
        eval_interval = self.tcfg.get("eval_interval", self.log_interval * 5)  # Less frequent eval
        
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            it = self.train_loader
            it = tqdm(it, desc=f"Epoch {epoch}", disable=not self.acc.is_main_process)
            
            # Skip batches if resuming mid-epoch (only on first iteration)
            if not resume_skip_done and self.current_step > 0:
                total_batches_consumed = self.current_step * gradient_accumulation_steps
                batches_to_skip = total_batches_consumed % steps_per_epoch
                if batches_to_skip > 0:
                    self.acc.print(f"[resume] Skipping {batches_to_skip} batches (step {self.current_step} x {gradient_accumulation_steps} grad_accum) in epoch {epoch}")
                    for skip_idx, _ in enumerate(it):
                        if skip_idx >= batches_to_skip - 1:
                            break
                resume_skip_done = True
            
            # Normal training loop with gradient accumulation
            for batch_idx, (x, y) in enumerate(it):
                # Use accumulate context manager for gradient accumulation
                with self.acc.accumulate(self.model):
                    output = self.model(input_ids=x, targets=y)
                    loss = output.loss
                    self.acc.backward(loss)
                    
                    # Only step optimizer when we've accumulated enough gradients
                    if self.acc.sync_gradients:
                        # Optional: gradient clipping for stability
                        max_grad_norm = self.tcfg.get("max_grad_norm", None)
                        if max_grad_norm is not None:
                            self.acc.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.current_step += 1
                        
                        # Log only on actual optimization steps
                        if self.current_step % self.log_interval == 0:
                            self._log(epoch, self.current_step, loss)
                        
                        # Eval less frequently and only on optimization steps
                        if self.current_step % eval_interval == 0:
                            if self.eval_loader is not None and self.acc.is_main_process:
                                val = self._evaluate(current_step=self.current_step, max_steps=self.cfg.training.get("eval_max_steps", 50))
                                if val is not None:
                                    self.acc.print(f"[eval] step {self.current_step} " + " ".join([f"{k}={v:.4f}" for k, v in val.items() if isinstance(v, float)]))
                                    self.acc.log({f"eval/{k}": v for k, v in val.items() if 'num' not in k}, step=self.current_step)
                        
                        if self.current_step % self.save_interval == 0:
                            self._save_ckpt(self.current_step)
            
            # Recreate dataloader for next epoch with deterministic seed
            self.train_loader = create_dataloader(
                txt_files=self.train_files,
                tokenizer=self.tok,
                batch_size=self.tcfg.batch_size,
                seq_len=self.mcfg.block_size,
                num_workers=self.dataloader_kwargs["num_workers"],
                shuffle=False,
                cache_size=self.tcfg.batch_size * self.mcfg.block_size * 64,
                dataset_shuffle=True,
                num_shards=self.dcfg.get("num_shards", None),
                prefetch_factor=self.dataloader_kwargs["prefetch_factor"],
                persistent_workers=self.dataloader_kwargs["persistent_workers"],
                seed=self._data_seed + epoch + 1,
            )
            self.train_loader = self.acc.prepare(self.train_loader)
        self._final_ckpt(self.current_step)
        self.acc.end_training()

    @torch.no_grad()
    def _evaluate(self, current_step: int, max_steps: int | None = None):
        metrics = {}
        if self.eval_loader is not None:
            self.model.eval()
            total_loss, total_entropy, n = 0.0, 0.0, 0
            for i, (x, y) in enumerate(self.eval_loader):
                output = self.model(input_ids=x, targets=y)
                total_loss += output.loss.item()
                # per-token entropy from logits
                probs = torch.softmax(output.logits.view(-1, output.logits.size(-1)), dim=-1)
                entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item()
                total_entropy += entropy
                n += 1
                if max_steps and i+1 >= max_steps: break
            self.model.train()
            metrics['loss'] = total_loss / max(n, 1)
            metrics['entropy_per_token'] = total_entropy / max(n, 1)
        
        base_model = self.acc.unwrap_model(self.model)
        base_model.eval()
        if hasattr(base_model, "gradient_checkpointing_disable"):
            base_model.gradient_checkpointing_disable()
        
        # Create evaluators fresh with current model
        evaluator_configs = {
            "model": base_model, 
            "tokenizer": self.tok,
            "device": self.acc.device,
            "move_format": "uci",
            "batch_size": 32,
            "max_new_tokens": 20,
            "temperature": 1.0,
            "top_k": None
        }
        
        if not self.acc.is_main_process:
            return metrics

        test_data_dir = self.dcfg.get("test_data_dir", "/scratch/js15262/LLM-Pretraining/data")
        _eval_table = [
            (HumanGamesEvaluator, "chess/test/human_games_test_opening.parquet", "human_opening", 100),
            (HumanGamesEvaluator, "chess/test/human_games_test_capture.parquet", "human_capture", 100),
            (HumanGamesEvaluator, "chess/test/human_games_test_random.parquet", "human_random", 100),
            (HumanGamesEvaluator, "chess/test/human_games_test_promotion.parquet", "human_promotion", 100),
            (HumanGamesEvaluator, "chess/test/human_games_test_check.parquet", "human_check", 100),
            (PuzzlesEvaluator, "chess/test/puzzles_grandmaster.csv", "puzzles_grandmaster", 100),
            (PuzzlesEvaluator, "chess/test/puzzles_test.csv", "puzzles_test", 100),
            (RandomGamesEvaluator, "chess/test/random_games_100.parquet", "random_games", 100),
        ]

        eval_configs = []
        for eval_cls, rel_path, prefix, max_samples in _eval_table:
            file_path = os.path.join(test_data_dir, rel_path)
            if not os.path.exists(file_path):
                continue
            eval_configs.append({
                "evaluator": eval_cls(**evaluator_configs),
                "file": file_path,
                "prefix": prefix,
                "max_samples": max_samples,
            })

        if not eval_configs:
            self.acc.print(f"[eval] No test files found under {test_data_dir}, skipping eval rollouts")
            return metrics
        for config in eval_configs:
            pred_dir = f"{self.save_dir}/rollouts/validation/{config['prefix']}"
            os.makedirs(pred_dir, exist_ok=True)

            output_path = f"{pred_dir}/predictions_{current_step}.parquet"
            output_metrics_path = f"{pred_dir}/metrics.jsonl"
            eval_metrics = config["evaluator"].evaluate_and_save_predictions(
                config["file"],
                output_path=output_path,
                max_samples=100,
                verbose=False
            )
            import json
            record = {"step": current_step, "prefix": config["prefix"], **eval_metrics}
            with open(output_metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            metrics.update({
                f"{config['prefix']}_{k}": v
                for k, v in eval_metrics.items()
                if "num" not in k
            })
            self.acc.print(f"Saved predictions for {config['prefix']} to {output_path}")
        
        return metrics

    @staticmethod
    def _board_from_pgn(state_text: str):
        """Parse a PGN movetext string into a chess.Board."""
        import chess, chess.pgn, io
        state_text = state_text.strip()
        try:
            return chess.Board(state_text)  # try FEN
        except Exception:
            pass
        game = chess.pgn.read_game(io.StringIO(state_text))
        return game.end().board() if game else chess.Board()

    @staticmethod
    def _move_to_lan(board, move):
        """Convert a chess.Move on the given board to LAN format (e.g. Pd2d4)."""
        import chess
        if board.is_kingside_castling(move):
            return "O-O"
        if board.is_queenside_castling(move):
            return "O-O-O"
        piece = board.piece_at(move.from_square)
        piece_char = piece.symbol().upper() if piece else "P"
        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)
        capture = "x" if board.is_capture(move) else ""
        promo = f"={chess.piece_symbol(move.promotion).upper()}" if move.promotion else ""
        return f"{piece_char}{from_sq}{capture}{to_sq}{promo}"

    @torch.no_grad()
    def _compute_test_entropy(self, evaluator, file_path, model, max_samples=100):
        """Compute move-level entropy and top-k probability mass on test positions.

        For each position, gets all legal moves, scores each via a batched
        teacher-forced forward pass, normalises to a distribution over legal
        moves, and computes entropy (nats) and top-k cumulative probability.

        Returns dict with keys: entropy, top1_mass, top3_mass, top5_mass
        """
        df = evaluator.load_data(file_path)
        state_col = evaluator.get_state_column()
        move_col = evaluator.get_move_column()
        df = df.dropna(subset=[state_col, move_col]).head(max_samples)
        if len(df) == 0:
            return None

        device = self.acc.device
        top_ks = [1, 3, 5]
        total_entropy, n_positions = 0.0, 0
        total_topk = {k: 0.0 for k in top_ks}

        for _, row in df.iterrows():
            state_text = str(row[state_col]).strip()
            try:
                board = self._board_from_pgn(state_text)
            except Exception:
                continue
            legal_moves = list(board.legal_moves)
            if len(legal_moves) == 0:
                continue

            # Convert each legal move to LAN and tokenize
            state_prefix_ids = self.tok.encode(state_text + " ")
            prefix_len = len(state_prefix_ids)
            sequences = []
            for mv in legal_moves:
                lan = self._move_to_lan(board, mv)
                full_ids = self.tok.encode(state_text + " " + lan)
                sequences.append(full_ids)

            # Pad and batch all candidates
            max_len = min(max(len(s) for s in sequences), self.mcfg.block_size)
            padded = torch.zeros(len(sequences), max_len, dtype=torch.long, device=device)
            for j, seq in enumerate(sequences):
                sl = min(len(seq), max_len)
                padded[j, :sl] = torch.tensor(seq[:sl], dtype=torch.long)

            # Single batched forward pass for all legal moves of this position
            output = model(input_ids=padded)
            logits = output.logits  # (num_legal_moves, T, V)
            log_probs = torch.log_softmax(logits, dim=-1)

            # Score each move: sum log P(token_t | prefix + tokens<t) over move tokens
            move_log_probs = []
            for j, seq in enumerate(sequences):
                sl = min(len(seq), max_len)
                ms = prefix_len - 1  # position whose logits predict the first move token
                if ms >= sl - 1:
                    move_log_probs.append(torch.tensor(0.0, device=device))
                    continue
                # gather log probs for each move token
                target_ids = padded[j, ms + 1 : sl]  # (move_len,)
                pred_log_probs = log_probs[j, ms : sl - 1]  # (move_len, V)
                token_lps = pred_log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
                move_log_probs.append(token_lps.sum())

            # Normalise over legal moves → proper distribution
            move_log_probs_t = torch.stack(move_log_probs)  # (num_legal_moves,)
            move_probs = torch.softmax(move_log_probs_t, dim=0)  # normalised

            # Entropy
            entropy = -(move_probs * torch.log(move_probs + 1e-9)).sum().item()
            total_entropy += entropy

            # Top-k probability mass
            sorted_probs, _ = move_probs.sort(descending=True)
            for k in top_ks:
                total_topk[k] += sorted_probs[:k].sum().item()

            n_positions += 1

        if n_positions == 0:
            return None
        result = {"entropy": total_entropy / n_positions}
        for k in top_ks:
            result[f"top{k}_mass"] = total_topk[k] / n_positions
        return result

    def _log(self, epoch, step, loss):
        if self.acc.is_main_process:
            lr = self.scheduler.get_last_lr()[0]
            now = time.time()
            elapsed = now - self._last_log_time
            steps_delta = step - self._last_log_step
            grad_accum = self.tcfg.get("gradient_accumulation_steps", 1)
            tokens_processed = steps_delta * self._tokens_per_batch * grad_accum
            tokens_per_sec = tokens_processed / max(elapsed, 1e-9)
            achieved_tflops = (tokens_per_sec * self._flops_per_token) / 1e12
            mfu = achieved_tflops / self._gpu_peak_tflops

            self._last_log_time = now
            self._last_log_step = step

            self.acc.print(
                f"epoch {epoch} step {step} loss={loss.item():.4f} lr={lr:.2e} "
                f"tok/s={tokens_per_sec:.0f} mfu={mfu:.3f}"
            )
            self.acc.log({
                "train/loss": loss.item(), "lr": lr, "epoch": epoch,
                "train/tokens_per_sec": tokens_per_sec,
                "train/tflops": achieved_tflops,
                "train/mfu": mfu,
            }, step=step)

    def _save_ckpt(self, step):
        out = self.save_dir / f"step{step}"
        if self.acc.is_main_process:
            out.mkdir(parents=True, exist_ok=True)
        self.acc.wait_for_everyone()
        self.acc.save_state(str(out))
        self._save_training_state(out)
        if self.acc.is_main_process:
            self.acc.print(f"[train] checkpoint saved → {out}")
            self._cleanup_old_checkpoints()

    def _final_ckpt(self, step):
        out = self.save_dir / f"final_step{step}"
        if self.acc.is_main_process:
            out.mkdir(parents=True, exist_ok=True)
        self.acc.wait_for_everyone()
        self.acc.save_state(str(out))
        self._save_training_state(out)
        if self.acc.is_main_process:
            self.acc.print(f"[train] final checkpoint → {out}")

    def _expand_txt_files(self, dcfg):
        """Return a sorted list[Path] of shard files from config."""
        from pathlib import Path
        def expand_one(p):
            p = Path(p)
            if p.is_dir():
                # non-recursive: only *.txt directly under the dir
                return sorted(p.glob("*.txt")) + sorted(p.glob("*.npy"))
            # allow simple glob strings too, e.g. '/data/chess/raw.*.txt'
            if any(ch in str(p) for ch in "*?[]"):
                return sorted(Path().glob(str(p)))
            # single file
            return [p] if p.exists() else []

        out = []
        if "txt_files" in dcfg and dcfg.txt_files:
            items = dcfg.txt_files
            if isinstance(items, (str, Path)):
                items = [items]
            for it in items:
                out.extend(expand_one(it))
        elif "txt_path" in dcfg and dcfg.txt_path:
            out.extend(expand_one(dcfg.txt_path))
        else:
            raise ValueError("Config must provide data.txt_files or data.txt_path")

        # de-dup & sort by name for stable shard order
        uniq = sorted({Path(p).resolve() for p in out}, key=lambda p: p.name)
        if not uniq:
            raise FileNotFoundError("No shards found from data paths.")
        return uniq

