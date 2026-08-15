"""
Trainer for mixed pretraining + SFT batches using the mixed dataloader.

Combines language-model pretraining shards with SFT JSON/JSONL data in a
single training loop. Uses the SFT loss (masked cross entropy) so SFT batches
can mask prompts while pretraining batches train on all tokens.
"""
import pathlib
import sys
import shutil
from typing import Optional, Dict, Any, List

from tqdm import tqdm

repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from accelerate import Accelerator
from .mixed_data_utils import create_mixed_dataloader
from .sft_data_utils import create_sft_dataloader
from .sft_loss import SFTLoss
from model.gpt2_model import GPT, GPTConfig
from llm_tokens.chess.tokenizer_factory import init_tokenizer
from .optim_sched import build_optimizer, build_scheduler
import torch


class MixedTrainer:
    """Trainer that alternates pretraining and SFT batches."""

    def __init__(self, cfg, run_config_path: Optional[str] = None):
        self.cfg = cfg
        self.run_cfg_path = run_config_path

        backend = cfg.logging.get("backend", "wandb")
        mixed_precision = cfg.training.get("mixed_precision", "no")
        gradient_accumulation_steps = cfg.training.get("gradient_accumulation_steps", 1)

        self.acc = Accelerator(
            log_with=("wandb" if backend == "wandb" else backend),
            mixed_precision=mixed_precision if mixed_precision != "no" else None,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        self._init_all()

    # ---------------- initialization helpers ----------------
    def _init_all(self):
        self.mcfg = self.cfg.model
        self.tcfg = self.cfg.training
        self.dcfg = self.cfg.data
        self.tokcfg = self.cfg.tokenizer

        self._init_tokenizer()
        self._init_data()
        self._init_model()
        self._init_optim()
        self._init_logging()

    def _init_tokenizer(self):
        self.tok = init_tokenizer(name=self.tokcfg.name, config=self.tokcfg)
        if hasattr(self.tok, "get_vocab_size"):
            self.vocab_size = int(self.tok.get_vocab_size())
        else:
            self.vocab_size = int(len(self.tok.get_vocab()))
        self.acc.print(f"[mixed] Vocab size: {self.vocab_size}")

    def _init_data(self):
        mix_cfg = self.dcfg.get("mixed", {})

        pretrain_specs = mix_cfg.get("pretrain_txt_files") or self.dcfg.get("pretrain_txt_files") or self.dcfg.get("txt_files")
        sft_specs = mix_cfg.get("sft_files") or self.dcfg.get("sft_files")

        if not pretrain_specs:
            raise ValueError("MixedTrainer requires data.mixed.pretrain_txt_files (or data.pretrain_txt_files)")
        if not sft_specs:
            raise ValueError("MixedTrainer requires data.mixed.sft_files (or data.sft_files)")

        pretrain_files = self._expand_txt_files({"txt_files": pretrain_specs})
        sft_files = self._get_sft_files(sft_specs)

        num_workers = int(self.tcfg.get("num_workers", 4))
        sft_num_workers = int(mix_cfg.get("sft_num_workers", self.tcfg.get("sft_num_workers", num_workers)))

        self.train_loader = create_mixed_dataloader(
            pretrain_txt_files=[str(p) for p in pretrain_files],
            sft_files=sft_files,
            tokenizer=self.tok,
            seq_len=self.mcfg.block_size,
            batch_size=self.tcfg.batch_size,
            sft_batch_size=mix_cfg.get("sft_batch_size"),
            pretrain_fraction=mix_cfg.get("pretrain_fraction", 0.5),
            total_batches=mix_cfg.get("total_batches"),
            num_workers=num_workers,
            sft_num_workers=sft_num_workers,
            cache_size=mix_cfg.get("cache_size", self.tcfg.batch_size * self.mcfg.block_size * 64),
            dataset_shuffle=mix_cfg.get("dataset_shuffle", True),
            num_shards=self.dcfg.get("num_shards", None),
            prefetch_factor=self.tcfg.get("prefetch_factor", 2 if num_workers > 0 else None),
            persistent_workers=self.tcfg.get("persistent_workers", num_workers > 0),
            sft_prefetch_factor=mix_cfg.get("sft_prefetch_factor", 2 if sft_num_workers > 0 else None),
            sft_persistent_workers=mix_cfg.get("sft_persistent_workers", sft_num_workers > 0),
            sft_mask_prompt=mix_cfg.get("sft_mask_prompt", True),
            sft_pad_token_id=mix_cfg.get("sft_pad_token_id", None),
            sft_cot_field=mix_cfg.get("sft_cot_field", "cot_format"),
            sft_prompt_field=mix_cfg.get("sft_prompt_field", "pgn"),
            seed=self.tcfg.get("seed", None),
        )

        eval_sft_files = self.dcfg.get("eval_sft_files", [])
        self.eval_loader = None
        if eval_sft_files:
            eval_sft_files = self._get_sft_files(eval_sft_files)
            self.eval_loader = create_sft_dataloader(
                data_files=eval_sft_files,
                tokenizer=self.tok,
                batch_size=self.tcfg.get("eval_batch_size", self.tcfg.batch_size),
                seq_len=self.mcfg.block_size,
                num_workers=0,
                shuffle=False,
                mask_prompt=mix_cfg.get("sft_mask_prompt", True),
                cot_field=mix_cfg.get("sft_cot_field", "cot_format"),
                prompt_field=mix_cfg.get("sft_prompt_field", "pgn"),
            )

    def _init_model(self):
        self.model = GPT(
            GPTConfig(
                vocab_size=self.vocab_size,
                block_size=self.mcfg.block_size,
                n_layer=self.mcfg.n_layer,
                n_head=self.mcfg.n_head,
                n_embed=self.mcfg.n_embed,
                dropout=self.mcfg.dropout,
                mlp_type=self.mcfg.get("mlp_type", "default"),
            )
        )

        if "pretrained_weights" in self.tcfg and self.tcfg.pretrained_weights:
            self._load_pretrained_weights(self.tcfg.pretrained_weights)

        if self.tcfg.get("compile_model", False):
            compile_mode = self.tcfg.get("compile_mode", "reduce-overhead")
            self.acc.print(f"[mixed] Compiling model with mode={compile_mode}")
            try:
                if hasattr(torch, "compile"):
                    self.model = torch.compile(self.model, mode=compile_mode)
                    self.acc.print("[mixed] Model compiled successfully")
            except Exception as e:
                self.acc.print(f"[mixed] Model compilation failed: {e}")

        self.loss_fn = SFTLoss(
            ignore_index=-100,
            reduction="mean",
            label_smoothing=self.tcfg.get("label_smoothing", 0.0),
        )

    def _init_optim(self):
        self.optimizer = build_optimizer(
            self.model,
            {
                "name": self.tcfg.optimizer.name,
                "lr": self.tcfg.optimizer.lr,
                "weight_decay": self.tcfg.optimizer.get("weight_decay", 0.1),
                "betas": tuple(self.tcfg.optimizer.get("betas", (0.9, 0.95))),
                "exclude_bias_and_norm": True,
            },
        )

        self.epochs = self.tcfg.epochs
        steps_per_epoch = len(self.train_loader)
        self.scheduler = build_scheduler(
            self.optimizer,
            {
                "name": self.tcfg.scheduler.name,
                "eta_min": self.tcfg.scheduler.get("eta_min", 0.0),
                "warmup_steps": self.tcfg.scheduler.warmup_steps,
            },
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
        )

        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.scheduler,
            self.eval_loader,
        ) = self.acc.prepare(self.model, self.optimizer, self.train_loader, self.scheduler, self.eval_loader)

    def _init_logging(self):
        run_name = self._auto_run_name(self.cfg)
        self.save_dir = pathlib.Path(self.tcfg.get("save_dir", "checkpoints/mixed")) / run_name
        self.save_interval = self.tcfg.get("save_interval", 1000)
        self.log_interval = self.tcfg.get("log_interval", 50)
        self.eval_interval = self.tcfg.get("eval_interval", self.log_interval * 5)
        self.resume_path = self.tcfg.get("resume", None)
        self.smoke_test = bool(self.tcfg.get("smoke_test", False))

        self.current_step = 0
        self.current_epoch = 0

        if self.acc.is_main_process:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        project_name = self.cfg.logging.get("project", "llm-training")
        self.acc.print(f"[mixed] Initing trackers: project={project_name} run={run_name}")
        self.acc.init_trackers(project_name, init_kwargs=self._tracker_init_kwargs(run_name, self._tracker_cfg()))
        if self.acc.is_main_process:
            import wandb

            if wandb.run is not None:
                wandb.run.name = run_name
                self.acc.print(f"[wandb] effective name: {wandb.run.name} (id={wandb.run.id})")

        if self.acc.is_main_process:
            self._snapshot_config()

        if self.resume_path:
            self.acc.load_state(self.resume_path)
            self._load_training_state(self.resume_path)
            self.acc.print(f"[mixed] Resumed from {self.resume_path} at step {self.current_step}")

    # ---------------- utility helpers ----------------
    def _tracker_cfg(self):
        cfg = self.cfg
        kinds = self.acc.log_with if isinstance(self.acc.log_with, (list, tuple)) else [self.acc.log_with]
        if "wandb" not in kinds:
            return {}

        lg = cfg.logging
        return {
            "wandb": {
                "entity": lg.get("entity", None),
                "name": self._auto_run_name(cfg),
                "notes": lg.get("notes", None),
                "tags": lg.get("tags", []) + ["mixed"],
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
            self.acc.print(f"[mixed] Config snapshot failed: {e}")

    def _save_training_state(self, checkpoint_dir):
        import json

        if self.acc.is_main_process:
            state_file = pathlib.Path(checkpoint_dir) / "training_state.json"
            state = {
                "step": self.current_step,
                "epoch": self.current_epoch,
            }
            with open(state_file, "w") as f:
                json.dump(state, f)

    def _load_training_state(self, checkpoint_dir):
        import json

        state_file = pathlib.Path(checkpoint_dir) / "training_state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                state = json.load(f)
            self.current_step = state.get("step", 0)
            self.current_epoch = state.get("epoch", 0)
        else:
            self.acc.print("[mixed] No training_state.json found, starting from step 0")

    def _auto_run_name(self, cfg):
        import time

        name = cfg.logging.get("run_name", None)
        if name:
            return name
        ts = time.strftime("%Y%m%d-%H%M%S")
        return f"mixed-{ts}"

    def _tracker_init_kwargs(self, run_name: str, cfg: Dict[str, Any]):
        lg = self.cfg.logging
        kinds = self.acc.log_with if isinstance(self.acc.log_with, (list, tuple)) else [self.acc.log_with]
        if "wandb" not in kinds:
            return {}
        return {
            "wandb": {
                "entity": lg.get("entity", None),
                "name": run_name,
                "notes": lg.get("notes", None),
                "tags": lg.get("tags", []) + ["mixed"],
                "config": cfg,
            }
        }

    def _expand_txt_files(self, dcfg):
        from pathlib import Path

        def expand_one(p):
            p = Path(p)
            if p.is_dir():
                return sorted(p.glob("*.txt")) + sorted(p.glob("*.npy"))
            if any(ch in str(p) for ch in "*?[]"):
                return sorted(Path().glob(str(p)))
            return [p] if p.exists() else []

        out: List[pathlib.Path] = []
        if "txt_files" in dcfg and dcfg["txt_files"]:
            items = dcfg["txt_files"]
            if isinstance(items, (str, pathlib.Path)):
                items = [items]
            for it in items:
                out.extend(expand_one(it))
        elif "txt_path" in dcfg and dcfg["txt_path"]:
            out.extend(expand_one(dcfg["txt_path"]))
        else:
            raise ValueError("Config must provide txt_files or txt_path for pretraining data")

        uniq = sorted({Path(p).resolve() for p in out}, key=lambda p: p.name)
        if not uniq:
            raise FileNotFoundError("No shards found from data paths.")
        return uniq

    def _get_sft_files(self, file_specs):
        from pathlib import Path
        import glob as glob_module

        files = []
        if isinstance(file_specs, str):
            file_specs = [file_specs]

        for spec in file_specs:
            p = Path(spec)
            if p.is_file():
                files.append(str(p))
            elif p.is_dir():
                files.extend([str(f) for f in p.glob("*.jsonl")])
                files.extend([str(f) for f in p.glob("*.json")])
            else:
                files.extend(glob_module.glob(str(p)))

        return sorted(set(files))

    def _load_pretrained_weights(self, path: str):
        from pathlib import Path
        import torch.nn.functional as F

        path = Path(path)

        if path.is_dir():
            model_path = path / "model.safetensors"
            if not model_path.exists():
                model_path = path / "pytorch_model.bin"
            if not model_path.exists():
                alt_paths = list(path.glob("*.safetensors")) + list(path.glob("*.bin")) + list(path.glob("*.pt")) + list(path.glob("*.pth"))
                if alt_paths:
                    model_path = alt_paths[0]
                else:
                    raise FileNotFoundError(f"No model weights found in {path}")
        else:
            model_path = path

        self.acc.print(f"[mixed] Loading pretrained weights from {model_path}")

        if str(model_path).endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(model_path)
        else:
            state_dict = torch.load(model_path, map_location="cpu")

        if "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]

        model_state = self.model.state_dict()
        resized_embeddings = False

        if self.vocab_size > model_state["transformer.wte.weight"].shape[0]:
            old_size = model_state["transformer.wte.weight"].shape[0]
            new_embed = F.pad(model_state["transformer.wte.weight"], (0, 0, 0, self.vocab_size - old_size))
            state_dict["transformer.wte.weight"] = new_embed
            state_dict["lm_head.weight"] = F.pad(state_dict.get("lm_head.weight", new_embed), (0, 0, 0, self.vocab_size - old_size))
            resized_embeddings = True

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        self.acc.print(f"[mixed] Loaded pretrained weights with missing keys: {missing}, unexpected keys: {unexpected}")
        if resized_embeddings:
            self.acc.print("[mixed] Resized token embeddings to match tokenizer vocab size")

    # ---------------- training & evaluation ----------------
    def train(self):
        self.model.train()

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            iterator = tqdm(self.train_loader, desc=f"Epoch {epoch}", disable=not self.acc.is_main_process)

            for batch_idx, batch in enumerate(iterator):
                with self.acc.accumulate(self.model):
                    input_ids = batch["input_ids"]
                    labels = batch["labels"]
                    attention_mask = batch.get("attention_mask", None)

                    output = self.model(input_ids=input_ids)
                    logits = output.logits if hasattr(output, "logits") else output

                    loss, metrics = self.loss_fn(logits=logits, labels=labels, attention_mask=attention_mask, return_metrics=True)

                    self.acc.backward(loss)

                    if self.acc.sync_gradients:
                        max_grad_norm = self.tcfg.get("max_grad_norm", None)
                        if max_grad_norm is not None:
                            self.acc.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.current_step += 1

                        if self.current_step % self.log_interval == 0:
                            self._log(epoch, self.current_step, metrics, batch.get("data_type", "mixed"))

                        if self.eval_loader is not None and self.current_step % self.eval_interval == 0:
                            eval_metrics = self._evaluate()
                            if eval_metrics and self.acc.is_main_process:
                                self.acc.print(
                                    f"[mixed eval] step {self.current_step} "
                                    + " ".join([f"{k}={v:.4f}" for k, v in eval_metrics.items()])
                                )
                                self.acc.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=self.current_step)

                        if self.current_step % self.save_interval == 0:
                            self._save_ckpt(self.current_step)

        self._final_ckpt(self.current_step)
        self.acc.end_training()

    @torch.no_grad()
    def _evaluate(self, max_steps: Optional[int] = 50):
        metrics = {}
        if self.eval_loader is None:
            return metrics

        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_tokens = 0
        n_batches = 0

        for i, batch in enumerate(self.eval_loader):
            if max_steps and i >= max_steps:
                break

            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch.get("attention_mask", None)

            output = self.model(input_ids=input_ids)
            logits = output.logits if hasattr(output, "logits") else output

            loss, batch_metrics = self.loss_fn(logits=logits, labels=labels, attention_mask=attention_mask, return_metrics=True)

            total_loss += batch_metrics["loss"]
            total_accuracy += batch_metrics["accuracy"]
            total_tokens += batch_metrics["num_valid_tokens"]
            n_batches += 1

        if n_batches > 0:
            metrics["loss"] = total_loss / n_batches
            metrics["accuracy"] = total_accuracy / n_batches
            metrics["num_tokens"] = total_tokens

        self.model.train()
        return metrics

    def _log(self, epoch, step, metrics: Dict[str, Any], data_type: str):
        lr = self.scheduler.get_last_lr()[0]
        log_payload = {
            "train/loss": metrics.get("loss", 0.0),
            "train/accuracy": metrics.get("accuracy", 0.0),
            "lr": lr,
            "epoch": epoch,
            "data_type/pretrain": 1.0 if data_type == "pretrain" else 0.0,
        }
        self.acc.print(
            f"epoch {epoch} step {step} loss={log_payload['train/loss']:.4f} "
            f"acc={log_payload['train/accuracy']:.4f} lr={lr:.2e} type={data_type}"
        )
        self.acc.log(log_payload, step=step)

    def _save_ckpt(self, step):
        if self.acc.is_main_process:
            out = self.save_dir / f"step{step}"
            out.mkdir(parents=True, exist_ok=True)
            self.acc.save_state(str(out))
            self._save_training_state(out)
            self.acc.print(f"[mixed] checkpoint saved → {out}")

    def _final_ckpt(self, step):
        if self.acc.is_main_process:
            out = self.save_dir / f"final_step{step}"
            out.mkdir(parents=True, exist_ok=True)
            self.acc.save_state(str(out))
            self._save_training_state(out)
            self.acc.print(f"[mixed] final checkpoint → {out}")

