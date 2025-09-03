import time
from typing import Any, Dict, Optional

from transformers import TrainerCallback, TrainerState, TrainingArguments


class ProgressCallback(TrainerCallback):
    """Logs training progress (percent, ETA, steps/sec, samples/sec).

    Uses args.max_steps when available (e.g., in streaming) to compute percent/ETA.
    Falls back to state.max_steps if set by HF Trainer.
    """

    def __init__(self, log_interval_steps: Optional[int] = None) -> None:
        self.start_time: float = 0.0
        self.total_steps: Optional[int] = None
        self.last_logged_step: int = -1
        self.log_interval_steps = log_interval_steps

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control, **kwargs):
        self.start_time = time.time()
        # Prefer explicit max_steps if provided (common in streaming)
        if getattr(args, "max_steps", 0):
            self.total_steps = int(args.max_steps)
        elif getattr(state, "max_steps", 0):
            # HF sets this when dataset length is known
            self.total_steps = int(state.max_steps)
        else:
            self.total_steps = None
        return control

    def on_log(self, args: TrainingArguments, state: TrainerState, control, logs: Dict[str, Any], **kwargs):
        # Respect main process only
        if hasattr(args, "local_rank") and args.local_rank not in (-1, 0):
            return control

        step = int(state.global_step)
        if self.log_interval_steps and step - self.last_logged_step < self.log_interval_steps:
            return control

        elapsed = max(1e-6, time.time() - self.start_time)
        steps_per_sec = step / elapsed if step > 0 else 0.0

        # Train batch size per optimizer step; samples/sec is approximate
        try:
            batch_per_device = args.per_device_train_batch_size
            world = max(1, args.world_size if hasattr(args, "world_size") else 1)
            eff_bs = batch_per_device * world
            # Account for gradient accumulation at optimizer step granularity
            eff_bs *= max(1, args.gradient_accumulation_steps)
        except Exception:
            eff_bs = None

        samples_per_sec = steps_per_sec * eff_bs if eff_bs else None

        if self.total_steps and self.total_steps > 0:
            remaining = max(0, self.total_steps - step)
            eta_sec = remaining / steps_per_sec if steps_per_sec > 0 else None
            percent = 100.0 * (step / self.total_steps)
        else:
            eta_sec = None
            percent = None

        progress_logs: Dict[str, Any] = {
            "progress/step": step,
            "progress/steps_per_sec": steps_per_sec,
        }
        if samples_per_sec is not None:
            progress_logs["progress/samples_per_sec"] = samples_per_sec
        if percent is not None:
            progress_logs["progress/percent"] = percent
        if eta_sec is not None:
            progress_logs["progress/eta_minutes"] = eta_sec / 60.0

        # Merge with HF's logs and emit via Trainer
        # The Trainer handles honoring logging_steps and report_to backends
        try:
            from transformers.trainer import Trainer

            trainer: Optional[Trainer] = kwargs.get("model")  # not available here
        except Exception:
            trainer = None

        # Use state.log_history mechanism by augmenting logs via control flow
        # The safest path: rely on the Trainer's logger via args.run_name/report_to
        # We can attach to the running Trainer via kwargs if provided under 'trainer'
        t = kwargs.get("trainer")
        if t is not None:
            t.log(progress_logs)
        else:
            # Fallback: attach to 'logs' dict (some UIs merge on_log logs)
            logs.update(progress_logs)

        self.last_logged_step = step
        return control

