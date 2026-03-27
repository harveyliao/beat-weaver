"""Training loop with mixed-precision, checkpointing, and TensorBoard logging."""

from __future__ import annotations

import json
import logging
import statistics
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from beat_weaver.model.config import ModelConfig
from beat_weaver.model.dataset import BeatSaberDataset, build_weighted_sampler, collate_fn
from beat_weaver.model.tokenizer import LEFT_BASE, LEFT_COUNT, PAD, RIGHT_BASE, RIGHT_COUNT
from beat_weaver.model.transformer import BeatWeaverModel

logger = logging.getLogger(__name__)


@dataclass
class TrainDebugOptions:
    """Optional knobs for short-run profiling and benchmarking."""

    max_steps: int | None = None
    profile: bool = False
    profile_wait: int = 5
    profile_warmup: int = 5
    profile_active: int = 10
    profile_log_interval: int = 20
    skip_validation: bool = False


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer, config: ModelConfig, steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Cosine LR schedule with linear warmup.

    steps_per_epoch should be the number of *optimizer steps* per epoch
    (i.e. len(train_loader) // gradient_accumulation_steps), not the raw
    batch count.
    """
    total_steps = config.max_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
        return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _color_balance_loss(logits: torch.Tensor) -> torch.Tensor:
    """Penalize deviation from 50/50 LEFT/RIGHT token probability.

    Only considers positions where a note token (LEFT or RIGHT) is likely.
    """
    logits_fp32 = logits.float()
    left_logits = logits_fp32[:, :, LEFT_BASE : LEFT_BASE + LEFT_COUNT]
    right_logits = logits_fp32[:, :, RIGHT_BASE : RIGHT_BASE + RIGHT_COUNT]

    left_log_mass = torch.logsumexp(left_logits, dim=-1)
    right_log_mass = torch.logsumexp(right_logits, dim=-1)
    note_log_mass = torch.logaddexp(left_log_mass, right_log_mass)
    total_log_mass = torch.logsumexp(logits_fp32, dim=-1)

    # Soft mask instead of boolean indexing to avoid extra CUDA gather/sync work.
    note_weight = (note_log_mass - total_log_mass).exp()
    active_weight = torch.where(note_weight > 0.1, note_weight, torch.zeros_like(note_weight))
    denom = active_weight.sum()
    if denom.item() == 0:
        return logits_fp32.new_zeros(())

    left_ratio = (left_log_mass - note_log_mass).exp()
    imbalance = (left_ratio - 0.5).square()
    return (imbalance * active_weight).sum() / denom


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summarize_phase_times(samples: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for name, values in samples.items():
        if not values:
            continue
        ordered = sorted(values)
        summary[name] = {
            "mean_ms": round(statistics.fmean(values) * 1000, 3),
            "p50_ms": round(statistics.median(values) * 1000, 3),
            "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] * 1000, 3),
            "max_ms": round(max(values) * 1000, 3),
        }
    return summary


def _log_phase_summary(summary: dict[str, dict[str, float]], steps: int) -> None:
    if not summary:
        return
    parts = []
    for phase, stats in summary.items():
        parts.append(
            f"{phase}=mean {stats['mean_ms']:.1f}ms p95 {stats['p95_ms']:.1f}ms max {stats['max_ms']:.1f}ms"
        )
    logger.info("Step timing summary over %d steps: %s", steps, "; ".join(parts))


class Trainer:
    """Wraps training state and provides train/validate methods."""

    def __init__(
        self,
        model: BeatWeaverModel,
        config: ModelConfig,
        output_dir: Path,
        device: torch.device | None = None,
    ):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device or _get_device()

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=PAD,
            label_smoothing=config.label_smoothing,
        )
        self.scaler = torch.amp.GradScaler(enabled=self.device.type == "cuda")
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))

        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(
        self,
        dataloader: DataLoader,
        *,
        max_steps: int | None = None,
        profile_log_interval: int = 50,
        profiler: torch.profiler.profile | None = None,
    ) -> tuple[float, int, dict[str, dict[str, float]]]:
        """Run one training epoch. Returns average loss, steps run, and timing summary."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        accum_steps = self.config.gradient_accumulation_steps
        phase_times = {
            "batch_fetch": [],
            "host_to_device": [],
            "forward_loss": [],
            "backward": [],
            "optimizer": [],
            "step_total": [],
        }

        self.optimizer.zero_grad()
        data_iter = iter(dataloader)
        batch_idx = 0
        while True:
            if max_steps is not None and n_batches >= max_steps:
                break
            fetch_started = time.perf_counter()
            try:
                mel, mel_mask, tokens, token_mask = next(data_iter)
            except StopIteration:
                break
            phase_times["batch_fetch"].append(time.perf_counter() - fetch_started)

            step_started = time.perf_counter()
            transfer_started = time.perf_counter()
            mel = mel.to(self.device)
            mel_mask = mel_mask.to(self.device)
            tokens = tokens.to(self.device)
            token_mask = token_mask.to(self.device)
            _sync_cuda(self.device)
            phase_times["host_to_device"].append(time.perf_counter() - transfer_started)

            # Teacher forcing: input is tokens[:-1], target is tokens[1:]
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            input_mask = token_mask[:, :-1]

            forward_started = time.perf_counter()
            with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                logits = self.model(mel, input_tokens, mel_mask, input_mask)
                # logits: (batch, seq_len-1, vocab_size)
                # target: (batch, seq_len-1)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_tokens.reshape(-1),
                )
                # Color balance auxiliary loss
                if self.config.color_balance_weight > 0:
                    loss = loss + self.config.color_balance_weight * _color_balance_loss(logits)
                loss = loss / accum_steps  # Scale for accumulation
            _sync_cuda(self.device)
            phase_times["forward_loss"].append(time.perf_counter() - forward_started)

            backward_started = time.perf_counter()
            self.scaler.scale(loss).backward()
            _sync_cuda(self.device)
            phase_times["backward"].append(time.perf_counter() - backward_started)

            optimizer_duration = 0.0
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                optimizer_started = time.perf_counter()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if hasattr(self, "scheduler"):
                    self.scheduler.step()
                _sync_cuda(self.device)
                optimizer_duration = time.perf_counter() - optimizer_started
            phase_times["optimizer"].append(optimizer_duration)

            total_loss += loss.item() * accum_steps  # Unscale for logging
            n_batches += 1
            self.global_step += 1
            phase_times["step_total"].append(time.perf_counter() - step_started)
            if profiler is not None:
                profiler.step()

            # Log every N steps
            if self.global_step % profile_log_interval == 0:
                self.writer.add_scalar("train/loss_step", loss.item() * accum_steps, self.global_step)
                lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("train/lr", lr, self.global_step)
                if self.device.type == "cuda":
                    self.writer.add_scalar(
                        "train/gpu_memory_allocated_mb",
                        torch.cuda.memory_allocated(self.device) / (1024 * 1024),
                        self.global_step,
                    )
                    self.writer.add_scalar(
                        "train/gpu_memory_reserved_mb",
                        torch.cuda.memory_reserved(self.device) / (1024 * 1024),
                        self.global_step,
                    )

            batch_idx += 1

        avg_loss = total_loss / max(1, n_batches)
        return avg_loss, n_batches, _summarize_phase_times(phase_times)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        """Run validation. Returns dict of metrics."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        n_batches = 0

        for mel, mel_mask, tokens, token_mask in dataloader:
            mel = mel.to(self.device)
            mel_mask = mel_mask.to(self.device)
            tokens = tokens.to(self.device)
            token_mask = token_mask.to(self.device)

            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            input_mask = token_mask[:, :-1]
            target_mask = token_mask[:, 1:]

            logits = self.model(mel, input_tokens, mel_mask, input_mask)
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1),
            )
            total_loss += loss.item()
            n_batches += 1

            # Token accuracy (ignoring padding)
            preds = logits.argmax(dim=-1)
            mask = target_mask & (target_tokens != PAD)
            total_correct += (preds == target_tokens)[mask].sum().item()
            total_tokens += mask.sum().item()

        avg_loss = total_loss / max(1, n_batches)
        accuracy = total_correct / max(1, total_tokens)
        return {"val_loss": avg_loss, "val_token_accuracy": accuracy}

    def save_checkpoint(self, name: str) -> Path:
        """Save model + optimizer + scheduler + training state."""
        ckpt_dir = self.output_dir / "checkpoints" / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), ckpt_dir / "model.pt")
        torch.save(self.optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        if hasattr(self, "scheduler"):
            torch.save(self.scheduler.state_dict(), ckpt_dir / "scheduler.pt")
        torch.save(self.scaler.state_dict(), ckpt_dir / "scaler.pt")
        self.config.save(ckpt_dir / "config.json")

        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }
        (ckpt_dir / "training_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        return ckpt_dir

    def load_checkpoint(self, ckpt_dir: Path) -> None:
        """Resume from a checkpoint."""
        ckpt_dir = Path(ckpt_dir)
        self._resume_dir = ckpt_dir
        self.model.load_state_dict(
            torch.load(ckpt_dir / "model.pt", map_location=self.device, weights_only=True),
        )
        self.optimizer.load_state_dict(
            torch.load(ckpt_dir / "optimizer.pt", map_location=self.device, weights_only=True),
        )
        scaler_path = ckpt_dir / "scaler.pt"
        if scaler_path.exists():
            self.scaler.load_state_dict(
                torch.load(scaler_path, map_location=self.device, weights_only=True),
            )
            logger.info("Restored GradScaler state from %s", scaler_path)
        elif self.device.type == "cuda":
            # No scaler.pt — use conservative scale=1.0 to avoid overflow
            self.scaler = torch.amp.GradScaler(init_scale=1.0, growth_interval=1000)
            logger.info("No scaler.pt found; using conservative init_scale=1.0")
        state = json.loads((ckpt_dir / "training_state.json").read_text(encoding="utf-8"))
        self.epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.best_val_loss = state["best_val_loss"]

    def restore_scheduler(self) -> None:
        """Restore scheduler state if resuming and scheduler.pt exists.

        If no scheduler.pt is found, fast-forward the scheduler to the
        current global_step so the LR matches where training left off.
        """
        resume_dir = getattr(self, "_resume_dir", None)
        if resume_dir is None:
            return
        if not hasattr(self, "scheduler"):
            return
        scheduler_path = resume_dir / "scheduler.pt"
        if scheduler_path.exists():
            self.scheduler.load_state_dict(
                torch.load(scheduler_path, map_location=self.device, weights_only=True),
            )
            logger.info("Restored LR scheduler state from %s", scheduler_path)
        elif self.global_step > 0:
            # No scheduler.pt — fast-forward to match resumed global_step
            for _ in range(self.global_step):
                self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                "Fast-forwarded LR scheduler to step %d (lr=%.2e)", self.global_step, lr,
            )


def train(
    config: ModelConfig,
    train_dataset: BeatSaberDataset,
    val_dataset: BeatSaberDataset,
    output_dir: Path,
    resume_from: Path | None = None,
    debug: TrainDebugOptions | None = None,
) -> Path:
    """Main training entry point.

    Returns path to the best checkpoint directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug = debug or TrainDebugOptions()

    model = BeatWeaverModel(config)
    logger.info("Model parameters: %s", f"{model.count_parameters():,}")

    trainer = Trainer(model, config, output_dir)

    if resume_from:
        trainer.load_checkpoint(resume_from)
        logger.info("Resumed from %s (epoch %d)", resume_from, trainer.epoch)

    sampler = build_weighted_sampler(train_dataset, config.official_ratio)
    use_cuda = trainer.device.type == "cuda"
    # Windows spawn-based multiprocessing causes DataLoader worker deadlocks
    # between epochs with persistent_workers. Use num_workers=0 on Windows.
    num_workers = 0 if sys.platform == "win32" else (2 if use_cuda else 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,  # shuffle only when no weighted sampler
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )

    # Build scheduler after knowing steps_per_epoch.
    # The scheduler steps once per optimizer step, which happens every
    # gradient_accumulation_steps batches (not every batch).
    optimizer_steps_per_epoch = max(1, len(train_loader) // config.gradient_accumulation_steps)
    trainer.scheduler = _build_lr_scheduler(
        trainer.optimizer, config, optimizer_steps_per_epoch,
    )
    trainer.restore_scheduler()

    config.save(output_dir / "config.json")

    training_start = time.time()
    epoch_times: list[float] = []

    logger.info(
        "Training: %d train samples, %d val samples, %d batches/epoch, device=%s",
        len(train_dataset), len(val_dataset), len(train_loader), trainer.device,
    )
    if debug.max_steps is not None:
        logger.info(
            "Short-run mode enabled: max_steps=%d, profile=%s, skip_validation=%s",
            debug.max_steps, debug.profile, debug.skip_validation,
        )

    profiler_ctx = None
    if debug.profile:
        trace_dir = output_dir / "profiling"
        trace_dir.mkdir(parents=True, exist_ok=True)
        profiler_ctx = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                *( [torch.profiler.ProfilerActivity.CUDA] if trainer.device.type == "cuda" else [] ),
            ],
            schedule=torch.profiler.schedule(
                wait=debug.profile_wait,
                warmup=debug.profile_warmup,
                active=debug.profile_active,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

    total_profiled_steps = 0
    collected_phase_summaries: list[dict[str, dict[str, float]]] = []

    with profiler_ctx if profiler_ctx is not None else nullcontext() as maybe_profiler:
        profiler = maybe_profiler if profiler_ctx is not None else None
        for epoch in range(trainer.epoch, config.max_epochs):
            trainer.epoch = epoch
            t0 = time.time()
            remaining_steps = None
            if debug.max_steps is not None:
                remaining_steps = max(0, debug.max_steps - total_profiled_steps)
                if remaining_steps == 0:
                    break

            train_loss, steps_run, phase_summary = trainer.train_epoch(
                train_loader,
                max_steps=remaining_steps,
                profile_log_interval=debug.profile_log_interval,
                profiler=profiler,
            )
            total_profiled_steps += steps_run
            if phase_summary:
                collected_phase_summaries.append(phase_summary)
                _log_phase_summary(phase_summary, steps_run)

            if debug.skip_validation and debug.max_steps is not None:
                val_metrics = {"val_loss": float("nan"), "val_token_accuracy": float("nan")}
            else:
                val_metrics = trainer.validate(val_loader)

            elapsed = time.time() - t0
            epoch_times.append(elapsed)
            total_elapsed = time.time() - training_start

            logger.info(
                "Epoch %d/%d (%.1fs, total %.0fs): train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                epoch + 1, config.max_epochs, elapsed, total_elapsed,
                train_loss, val_metrics["val_loss"], val_metrics["val_token_accuracy"],
            )

            # TensorBoard
            trainer.writer.add_scalar("train/loss_epoch", train_loss, epoch)
            trainer.writer.add_scalar("val/loss", val_metrics["val_loss"], epoch)
            trainer.writer.add_scalar("val/token_accuracy", val_metrics["val_token_accuracy"], epoch)
            trainer.writer.add_scalar("timing/epoch_seconds", elapsed, epoch)
            trainer.writer.add_scalar("timing/total_seconds", total_elapsed, epoch)

            # Checkpoint every epoch
            trainer.save_checkpoint(f"epoch_{epoch + 1:03d}")

            # Best model
            if debug.skip_validation and debug.max_steps is not None and trainer.best_val_loss == float("inf"):
                trainer.save_checkpoint("best")
                logger.info("Saved baseline checkpoint for profiling run without validation")
            elif val_metrics["val_loss"] < trainer.best_val_loss:
                trainer.best_val_loss = val_metrics["val_loss"]
                trainer.patience_counter = 0
                trainer.save_checkpoint("best")
                logger.info("New best model (val_loss=%.4f)", trainer.best_val_loss)
            elif not debug.skip_validation:
                trainer.patience_counter += 1
                if trainer.patience_counter >= config.early_stopping_patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

            if debug.max_steps is not None and total_profiled_steps >= debug.max_steps:
                logger.info("Reached profiling step limit (%d); stopping training run", debug.max_steps)
                break

    trainer.writer.close()

    # Write training summary
    total_time = time.time() - training_start
    epochs_completed = len(epoch_times)
    avg_epoch = sum(epoch_times) / max(1, epochs_completed)
    if total_profiled_steps > 0:
        samples_per_second = round((total_profiled_steps * config.batch_size) / max(total_time, 1e-8), 1)
    else:
        samples_per_second = round(len(train_dataset) / avg_epoch, 1)
    summary = {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "batches_per_epoch": len(train_loader),
        "batch_size": config.batch_size,
        "epochs_completed": epochs_completed,
        "steps_profiled": total_profiled_steps,
        "total_time_seconds": round(total_time, 1),
        "avg_epoch_seconds": round(avg_epoch, 1),
        "samples_per_second": samples_per_second,
        "best_val_loss": round(trainer.best_val_loss, 4),
        "device": str(trainer.device),
        "model_parameters": model.count_parameters(),
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if collected_phase_summaries:
        profiling_summary = {
            "steps_profiled": total_profiled_steps,
            "profile_enabled": debug.profile,
            "max_steps": debug.max_steps,
            "skip_validation": debug.skip_validation,
            "phase_summaries": collected_phase_summaries,
        }
        profiling_summary_path = output_dir / "profiling_summary.json"
        profiling_summary_path.write_text(json.dumps(profiling_summary, indent=2), encoding="utf-8")
    logger.info(
        "Training complete: %d epochs in %.0fs (avg %.1fs/epoch, %.1f samples/s)",
        epochs_completed, total_time, avg_epoch,
        len(train_dataset) / avg_epoch,
    )

    return output_dir / "checkpoints" / "best"
