#!/usr/bin/env python3
"""
Embed text and invert it back to text using a trained checkpoint.

Usage examples

1) Using local inverter checkpoint only (single-step inversion):
   python scripts/embed_and_reverse.py \
     --inverter ./saves/my-inverter-checkpoint \
     --text "A quick brown fox jumps over the lazy dog."

2) Using local inverter + corrector checkpoints (multi-step correction):
   python scripts/embed_and_reverse.py \
     --inverter ./saves/my-inverter-checkpoint \
     --corrector ./saves/my-corrector-checkpoint \
     --text "A quick brown fox jumps over the lazy dog." \
     --num-steps 10 --beam-width 0

3) Using built-in pretrained alias (requires internet to download once):
   python scripts/embed_and_reverse.py \
     --pretrained gtr-base \
     --text "Cornell Tech is in New York City."

Notes
- If you pass only an inverter checkpoint, the script performs a single
  inversion step using the inverter alone (no corrector refinement).
- If you pass both inverter and corrector checkpoints or use --pretrained,
  you can set --num-steps (>0) for recursive correction and optionally
  --beam-width (>0) to expand the search.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import torch

import vec2text
from vec2text.api import invert_embeddings
from vec2text.models.model_utils import device as default_device
from vec2text.trainers.inversion import InversionTrainer as _InversionTrainer
import transformers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed text and invert using checkpoints.")

    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--pretrained", choices=["text-embedding-ada-002", "gtr-base"],
                    help="Load built-in pretrained corrector (downloads from HF if needed).")
    p.add_argument("--inverter", type=str, default=None,
                   help="Path or HF id for an InversionModel checkpoint.")
    p.add_argument("--corrector", type=str, default=None,
                   help="Path or HF id for a CorrectorEncoderModel checkpoint.")

    p.add_argument("--text", type=str, default=None,
                   help="Text to embed and invert. If omitted, reads lines from stdin.")
    p.add_argument("--num-steps", type=int, default=0,
                   help="Number of recursive correction steps (requires corrector). 0 = single-step inversion only.")
    p.add_argument("--beam-width", type=int, default=0,
                   help="Sequence beam width for search (requires num-steps > 0).")
    p.add_argument("--max-length", type=int, default=128,
                   help="Max sequence length for the embedder tokenizer.")
    p.add_argument("--device", type=str, default=None,
                   help="Force device: cuda|mps|cpu. Default: auto-detect.")
    p.add_argument("--print-embedding", action="store_true",
                   help="Print embedding shape and a small preview.")

    return p.parse_args()


def read_texts(arg_text: Optional[str]) -> List[str]:
    if arg_text is not None:
        return [arg_text]
    # Read non-empty lines from stdin
    texts = [ln.strip() for ln in sys.stdin.read().splitlines() if ln.strip()]
    if not texts:
        raise SystemExit("No input text provided. Use --text or pipe lines on stdin.")
    return texts


def main() -> None:
    args = parse_args()
    texts = read_texts(args.text)

    # Device selection
    dev = default_device if args.device is None else torch.device(args.device)

    # Build a Corrector wrapper when possible; otherwise fall back to inverter-only.
    corrector = None
    inversion_trainer = None

    if args.pretrained:
        # Load a ready-to-use corrector (includes an internal inversion trainer)
        corrector = vec2text.api.load_pretrained_corrector(args.pretrained)
        corrector.model.to(dev)
        corrector.inversion_trainer.model.to(dev)
    else:
        if args.inverter is None:
            raise SystemExit("Provide --inverter or --pretrained.")

        if args.corrector is None and args.num_steps > 0:
            raise SystemExit("--num-steps > 0 requires --corrector (or use --pretrained).")

        # Load inverter only (single-step) or inverter+corrector (multi-step)
        # Load via Transformers from_pretrained so it works with local dirs or HF ids.
        inversion_model = vec2text.models.InversionModel.from_pretrained(args.inverter)
        inversion_model.to(dev)

        # Build a minimal InversionTrainer-like wrapper for generation and embedding calls
        # by reusing the code path in the existing Trainer implementation.
        # We borrow the data collator and other small bits from the training stack only when needed.
        inv_trainer = _InversionTrainer(
            model=inversion_model,
            args=None,
            train_dataset=None,
            eval_dataset=None,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer=inversion_model.tokenizer,
                label_pad_token_id=-100,
            ),
        )
        inversion_trainer = inv_trainer

        if args.corrector is not None:
            corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained(args.corrector)
            corrector = vec2text.api.load_corrector(inversion_model=inversion_model,
                                                    corrector_model=corrector_model)
            corrector.model.to(dev)
            corrector.inversion_trainer.model.to(dev)

    # Embed texts using the embedder associated with the inversion model
    if corrector is not None:
        tokenizer = corrector.embedder_tokenizer
        model = corrector.inversion_trainer.model
    else:
        # inverter-only mode
        tokenizer = inversion_trainer.model.embedder_tokenizer  # type: ignore
        model = inversion_trainer.model  # type: ignore

    enc = tokenizer(
        texts,
        return_tensors="pt",
        max_length=args.max_length,
        truncation=True,
        padding="max_length",
    ).to(dev)

    with torch.no_grad():
        emb = model.call_embedding_model(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
        )

    if args.print-embedding:
        # Print a small preview of the embedding vector(s)
        shape = tuple(emb.shape)
        preview = emb[0, :8].detach().cpu().tolist() if emb.ndim == 2 else []
        print(f"Embedding shape: {shape}")
        if preview:
            print(f"Embedding[0][:8]: {preview}")

    # Invert back to text
    if corrector is not None:
        steps = args.num_steps if args.num_steps > 0 else None
        texts_out = invert_embeddings(
            embeddings=emb,
            corrector=corrector,
            num_steps=steps,
            sequence_beam_width=args.beam_width,
        )
    else:
        # Single-step inversion with inverter only
        gen_kwargs = dict(min_length=1, max_length=args.max_length)
        with torch.no_grad():
            out_ids = inversion_trainer.generate(  # type: ignore
                inputs={"frozen_embeddings": emb},
                generation_kwargs=gen_kwargs,
            )
        texts_out = inversion_trainer.model.tokenizer.batch_decode(  # type: ignore
            out_ids, skip_special_tokens=True
        )

    # Print results
    for i, (inp, out) in enumerate(zip(texts, texts_out)):
        print(f"\nExample {i}:\n  Input:  {inp}\n  Output: {out}")


if __name__ == "__main__":
    main()
