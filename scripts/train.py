#!/usr/bin/env python3
"""
Cross‑platform training script for vec2text that can train an inverter
and/or a corrector for an arbitrary embedding model using the existing
infrastructure. It provides simple VRAM presets for low (6GiB+8GiB),
mid (16GiB), and high (24GiB) memory setups.

Usage examples

1) Train inverter only (auto-select precision, low-VRAM preset):
   python scripts/train.py \
     --stage inverter \
     --embedder gtr_base \
     --dataset msmarco \
     --max-seq-length 128 \
     --vram 6+8 \
     --output-dir ./saves/gtr-msmarco-inverter

2) Train corrector only given an already-trained inverter directory:
   python scripts/train.py \
     --stage corrector \
     --embedder gtr_base \
     --dataset msmarco \
     --max-seq-length 128 \
     --vram 16 \
     --corrector-from-pretrained ./saves/gtr-msmarco-inverter \
     --output-dir ./saves/gtr-msmarco-corrector

3) Train both (inverter then corrector, sharing preset):
   python scripts/train.py \
     --stage both \
     --embedder gtr_base \
     --dataset msmarco \
     --max-seq-length 128 \
     --vram 24 \
     --output-dir ./saves/gtr-msmarco-run

Notes
- For multi-GPU, launch with torchrun. Example (2 GPUs):
    torchrun --nproc_per_node=2 scripts/train.py ...
- On Windows, DeepSpeed is typically unavailable; the script gracefully
  avoids it and uses gradient accumulation + checkpointing instead.
- For best memory usage on smaller GPUs, the script can precompute and
  use frozen embeddings, then mock the embedder during training.
"""

from __future__ import annotations

import argparse
import os
import platform
from dataclasses import asdict
from typing import Dict, Optional, Tuple

import torch

from vec2text.experiments import experiment_from_args
from vec2text.run_args import DataArguments, ModelArguments, TrainingArguments


def bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        # Newer PyTorch exposes this helper
        return torch.cuda.is_bf16_supported()
    except AttributeError:
        # Fallback: assume Ampere+ supports bf16
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 8


def try_import_deepspeed() -> bool:
    try:
        import deepspeed  # noqa: F401

        return True
    except Exception:
        return False


def detect_num_gpus() -> int:
    try:
        return torch.cuda.device_count()
    except Exception:
        return 0


def precision_flags() -> Dict[str, bool]:
    # Prefer bf16 if supported; otherwise fp16 on CUDA; otherwise full fp32.
    if bf16_supported():
        return {"bf16": True, "fp16": False}
    if torch.cuda.is_available():
        return {"bf16": False, "fp16": True}
    return {"bf16": False, "fp16": False}


def vram_presets(vram: str, stage: str, use_deepspeed_if_available: bool) -> Dict:
    """
    Returns preset overrides for training and model args keyed as:
      - training: dict of TrainingArguments fields
      - model: dict of ModelArguments fields
    """

    # Defaults that are safe and reasonably fast; refined below per VRAM.
    common_training = dict(
        num_train_epochs=100,
        learning_rate=2e-4,
        warmup_steps=10_000,
        eval_steps=25_000,
        save_steps=10_000,
        logging_steps=400,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        group_by_length=True,
        save_total_limit=2,
        lr_scheduler_type="constant_with_warmup",
        evaluation_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
    )

    # Favor consistent effective batch size of ~32 tokens per update.
    # Precompute embeddings on low/mid VRAM to save memory and delete embedder.
    if vram == "6+8":
        training = dict(
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=8,
        )
        model = dict(
            use_frozen_embeddings_as_input=True,
        )
        extra = dict(
            mock_embedder=True,
            deepspeed=("vec2text/ds_config.json" if use_deepspeed_if_available else None),
        )
    elif vram == "16":
        training = dict(
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=2,
        )
        model = dict(
            use_frozen_embeddings_as_input=True,
        )
        extra = dict(
            mock_embedder=True,
            deepspeed=None,
        )
    elif vram == "24":
        training = dict(
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=1,
        )
        # On large VRAM, allow training without freezing embedder inputs if desired.
        model = dict(
            use_frozen_embeddings_as_input=False,
        )
        extra = dict(
            mock_embedder=False,
            deepspeed=None,
        )
    elif vram == "auto":
        # Heuristic based on total available GPU memory
        total_gb = 0
        for i in range(detect_num_gpus()):
            try:
                p = torch.cuda.get_device_properties(i)
                total_gb += int(p.total_memory / (1024**3))
            except Exception:
                pass
        if total_gb >= 24:
            return vram_presets("24", stage, use_deepspeed_if_available)
        elif total_gb >= 16:
            return vram_presets("16", stage, use_deepspeed_if_available)
        else:
            return vram_presets("6+8", stage, use_deepspeed_if_available)
    else:
        raise ValueError(f"Unknown vram preset '{vram}'")

    # Corrector typically fits similar or slightly larger batches; keep same preset.
    # If needed, minor stage-specific tweaks go here.
    if stage == "corrector":
        pass

    return {
        "training": {**common_training, **training, **precision_flags(), **extra},
        "model": {**model},
    }


def make_inversion_args(
    embedder: str,
    dataset: str,
    model_name: str,
    max_seq_length: int,
    output_dir: Optional[str],
    use_wandb: bool,
    vram: str,
    wandb_base_url: Optional[str],
    wandb_api_key: Optional[str],
    wandb_entity: Optional[str],
    wandb_project: Optional[str],
    streaming: bool,
    overwrite_output_dir: bool = False,
    max_steps: Optional[int] = None,
) -> Tuple[ModelArguments, DataArguments, TrainingArguments]:
    use_deepspeed = try_import_deepspeed() and (platform.system() != "Windows")
    preset = vram_presets(vram=vram, stage="inverter", use_deepspeed_if_available=use_deepspeed)

    margs = ModelArguments(
        model_name_or_path=model_name,
        embedder_model_name=embedder,
        max_seq_length=max_seq_length,
        use_frozen_embeddings_as_input=preset["model"].get("use_frozen_embeddings_as_input", False),
    )
    dargs = DataArguments(dataset_name=dataset, streaming=streaming)
    targs = TrainingArguments(
        output_dir=(output_dir or None),
        per_device_train_batch_size=preset["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=preset["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=preset["training"]["gradient_accumulation_steps"],
        num_train_epochs=preset["training"]["num_train_epochs"],
        learning_rate=preset["training"]["learning_rate"],
        warmup_steps=preset["training"]["warmup_steps"],
        eval_steps=preset["training"]["eval_steps"],
        save_steps=preset["training"]["save_steps"],
        logging_steps=preset["training"]["logging_steps"],
        gradient_checkpointing=preset["training"]["gradient_checkpointing"],
        ddp_find_unused_parameters=preset["training"]["ddp_find_unused_parameters"],
        remove_unused_columns=preset["training"]["remove_unused_columns"],
        group_by_length=preset["training"]["group_by_length"],
        save_total_limit=preset["training"]["save_total_limit"],
        lr_scheduler_type=preset["training"]["lr_scheduler_type"],
        evaluation_strategy=preset["training"]["evaluation_strategy"],
        logging_strategy=preset["training"]["logging_strategy"],
        save_strategy=preset["training"]["save_strategy"],
        bf16=preset["training"]["bf16"],
        fp16=preset["training"]["fp16"],
        use_wandb=use_wandb,
        wandb_base_url=wandb_base_url,
        wandb_api_key=wandb_api_key,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        experiment="inversion",
        deepspeed=preset["training"].get("deepspeed"),
        mock_embedder=preset["training"].get("mock_embedder", False),
        overwrite_output_dir=overwrite_output_dir,
    )

    # Streaming is incompatible with frozen embeddings + mock embedder.
    if streaming:
        margs.use_frozen_embeddings_as_input = False
        targs.mock_embedder = False
        # For IterableDataset without __len__, Transformers needs max_steps.
        if max_steps is not None:
            targs.max_steps = max_steps
        else:
            # Sensible default if user didn't provide one.
            # Large enough to accommodate warmup/eval intervals.
            if getattr(targs, "max_steps", -1) is None or targs.max_steps <= 0:
                targs.max_steps = 100_000

    # Windows tends to be more stable with single-process data loading.
    if platform.system() == "Windows":
        targs.dataloader_num_workers = 0

    return margs, dargs, targs


def make_corrector_args(
    embedder: str,
    dataset: str,
    model_name: str,
    max_seq_length: int,
    output_dir: Optional[str],
    use_wandb: bool,
    vram: str,
    corrector_from_pretrained: str,
    wandb_base_url: Optional[str],
    wandb_api_key: Optional[str],
    wandb_entity: Optional[str],
    wandb_project: Optional[str],
    streaming: bool,
    overwrite_output_dir: bool = False,
    max_steps: Optional[int] = None,
) -> Tuple[ModelArguments, DataArguments, TrainingArguments]:
    use_deepspeed = try_import_deepspeed() and (platform.system() != "Windows")
    preset = vram_presets(vram=vram, stage="corrector", use_deepspeed_if_available=use_deepspeed)

    margs = ModelArguments(
        model_name_or_path=model_name,
        embedder_model_name=embedder,
        max_seq_length=max_seq_length,
        use_frozen_embeddings_as_input=preset["model"].get("use_frozen_embeddings_as_input", False),
    )
    dargs = DataArguments(dataset_name=dataset, streaming=streaming)
    targs = TrainingArguments(
        output_dir=(output_dir or None),
        per_device_train_batch_size=preset["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=preset["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=preset["training"]["gradient_accumulation_steps"],
        num_train_epochs=preset["training"]["num_train_epochs"],
        learning_rate=preset["training"]["learning_rate"],
        warmup_steps=preset["training"]["warmup_steps"],
        eval_steps=preset["training"]["eval_steps"],
        save_steps=preset["training"]["save_steps"],
        logging_steps=preset["training"]["logging_steps"],
        gradient_checkpointing=preset["training"]["gradient_checkpointing"],
        ddp_find_unused_parameters=preset["training"]["ddp_find_unused_parameters"],
        remove_unused_columns=preset["training"]["remove_unused_columns"],
        group_by_length=preset["training"]["group_by_length"],
        save_total_limit=preset["training"]["save_total_limit"],
        lr_scheduler_type=preset["training"]["lr_scheduler_type"],
        evaluation_strategy=preset["training"]["evaluation_strategy"],
        logging_strategy=preset["training"]["logging_strategy"],
        save_strategy=preset["training"]["save_strategy"],
        bf16=preset["training"]["bf16"],
        fp16=preset["training"]["fp16"],
        use_wandb=use_wandb,
        wandb_base_url=wandb_base_url,
        wandb_api_key=wandb_api_key,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        experiment="corrector",
        deepspeed=preset["training"].get("deepspeed"),
        mock_embedder=preset["training"].get("mock_embedder", False),
        corrector_model_from_pretrained=corrector_from_pretrained,
        overwrite_output_dir=overwrite_output_dir,
    )

    if streaming:
        margs.use_frozen_embeddings_as_input = False
        targs.mock_embedder = False
        if max_steps is not None:
            targs.max_steps = max_steps
        else:
            if getattr(targs, "max_steps", -1) is None or targs.max_steps <= 0:
                targs.max_steps = 100_000

    if platform.system() == "Windows":
        targs.dataloader_num_workers = 0

    return margs, dargs, targs


def run_experiment(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
    exp = experiment_from_args(model_args, data_args, training_args)
    return exp.run()


def main():
    parser = argparse.ArgumentParser(description="Train vec2text inverter and/or corrector with VRAM presets.")
    parser.add_argument("--stage", choices=["inverter", "corrector", "both"], default="both")
    parser.add_argument("--embedder", required=True, help="Embedding model name (see vec2text.models.model_utils)")
    parser.add_argument("--dataset", default="msmarco")
    parser.add_argument("--model", default="t5-base", help="Seq2seq backbone (default: t5-base)")
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--vram", choices=["6+8", "16", "24", "auto"], default="auto")
    parser.add_argument("--output-dir", default=None, help="Output directory. If not set, uses hashed default.")
    parser.add_argument(
        "--overwrite_output_dir",
        "--overwrite-output-dir",
        dest="overwrite_output_dir",
        action="store_true",
        help="Allow overwriting an existing output directory.",
    )
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    parser.set_defaults(use_wandb=False)
    # Optional custom W&B server settings
    parser.add_argument("--wandb-base-url", default=None, help="Custom W&B base URL, e.g., https://wandb.mycompany.com")
    parser.add_argument("--wandb-api-key", default=None, help="W&B API key to use (prefer env var)")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity/organization")
    parser.add_argument("--wandb-project", default=None, help="Override W&B project name")

    parser.add_argument("--streaming", action="store_true", help="Enable dataset streaming to avoid local storage.")
    parser.add_argument("--max-steps", type=int, default=None, help="When using --streaming, set total training steps (required by HF Trainer when dataset length is unknown).")

    # Corrector-specific
    parser.add_argument("--corrector-from-pretrained", default=None, help="Path/name of trained inverter to base corrector on (required if stage is 'corrector').")

    args = parser.parse_args()

    # Sanity: ensure nltk punkt is available for metrics during training/eval
    try:
        import nltk

        nltk.data.find("tokenizers/punkt")
    except Exception:
        try:
            import nltk

            nltk.download("punkt")
        except Exception:
            pass

    if args.stage in ("inverter", "both"):
        margs, dargs, targs = make_inversion_args(
            embedder=args.embedder,
            dataset=args.dataset,
            model_name=args.model,
            max_seq_length=args.max_seq_length,
            output_dir=(args.output_dir if args.stage == "inverter" else None),
            use_wandb=args.use_wandb,
            vram=args.vram,
            wandb_base_url=args.wandb_base_url,
            wandb_api_key=args.wandb_api_key,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            streaming=args.streaming,
            overwrite_output_dir=args.overwrite_output_dir,
            max_steps=args.max_steps,
        )
        print("[train] Inverter args:")
        print("  model_args:", asdict(margs))
        print("  data_args:", asdict(dargs))
        print("  training_args:", {k: v for k, v in asdict(targs).items() if k in ("output_dir", "per_device_train_batch_size", "per_device_eval_batch_size", "gradient_accumulation_steps", "bf16", "fp16", "learning_rate", "eval_steps", "save_steps", "deepspeed", "mock_embedder")})
        run_experiment(margs, dargs, targs)
        inversion_out_dir = targs.output_dir
    else:
        inversion_out_dir = None

    if args.stage in ("corrector", "both"):
        base = args.corrector_from_pretrained or inversion_out_dir
        if not base:
            raise ValueError("--corrector-from-pretrained must be provided when stage is 'corrector' and no prior inverter was just trained.")
        margs, dargs, targs = make_corrector_args(
            embedder=args.embedder,
            dataset=args.dataset,
            model_name=args.model,
            max_seq_length=args.max_seq_length,
            output_dir=args.output_dir if args.stage != "inverter" else None,
            use_wandb=args.use_wandb,
            vram=args.vram,
            corrector_from_pretrained=base,
            wandb_base_url=args.wandb_base_url,
            wandb_api_key=args.wandb_api_key,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            streaming=args.streaming,
            overwrite_output_dir=args.overwrite_output_dir,
            max_steps=args.max_steps,
        )
        print("[train] Corrector args:")
        print("  model_args:", asdict(margs))
        print("  data_args:", asdict(dargs))
        print("  training_args:", {k: v for k, v in asdict(targs).items() if k in ("output_dir", "per_device_train_batch_size", "per_device_eval_batch_size", "gradient_accumulation_steps", "bf16", "fp16", "learning_rate", "eval_steps", "save_steps", "deepspeed", "mock_embedder", "corrector_model_from_pretrained")})
        run_experiment(margs, dargs, targs)


if __name__ == "__main__":
    main()
