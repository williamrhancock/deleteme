#!/usr/bin/env python3
"""
Clean, production-safe fine-tuning script for Phi-3 / Mistral on RunPod using Unsloth.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# -------------------------------
# Minimal dependency validation
# -------------------------------

def ensure_import(pkg, pip_name=None):
    pip_name = pip_name or pkg
    try:
        __import__(pkg)
    except ImportError:
        print(f"⚠️  Missing package '{pip_name}', installing...")
        os.system(f"{sys.executable} -m pip install -q {pip_name}")
        __import__(pkg)

ensure_import("unsloth", "unsloth @ git+https://github.com/unslothai/unsloth.git")
ensure_import("datasets")
ensure_import("transformers")
ensure_import("trl")
ensure_import("bitsandbytes")

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer


# -------------------------------
# Utility functions
# -------------------------------

def load_jsonl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSONL file not found: {path}")

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if not all(k in obj for k in ("instruction", "output")):
                    raise ValueError(
                        f"Each JSONL entry must have 'instruction' and 'output'. Offending entry:\n{obj}"
                    )
                data.append(obj)
    return data


def phi3_format(instruction, output):
    return (
        f"<|user|>\n{instruction}<|end|>\n"
        f"<|assistant|>\n{output}<|end|>"
    )


def mistral_format(instruction, output):
    return f"[INST] {instruction} [/INST] {output} </s>"


def format_batch(batch, model_name):
    out = []
    fmt = phi3_format if "phi-3" in model_name.lower() else mistral_format
    for inst, outp in zip(batch["instruction"], batch["output"]):
        out.append(fmt(inst, outp))
    return {"text": out}


def detect_lora_modules(model):
    wanted = [
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj"
    ]
    found = []
    missing = []

    for w in wanted:
        if any(w in name for name, _ in model.named_modules()):
            found.append(w)
        else:
            missing.append(w)

    if missing:
        print(f"⚠️ Warning: Missing LoRA modules in this model: {missing}")
    return found


# -------------------------------
# Main training entrypoint
# -------------------------------

def main():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--val-path", required=True)
    parser.add_argument("--output-dir", required=True)

    # Model
    parser.add_argument("--model-name", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--max-seq-length", default=2048, type=int)

    # LoRA
    parser.add_argument("--lora-r", default=16, type=int)
    parser.add_argument("--lora-alpha", default=32, type=int)
    parser.add_argument("--lora-dropout", default=0.05, type=float)

    # Training
    parser.add_argument("--learning-rate", default=5e-5, type=float)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--gradient-accumulation-steps", default=8, type=int)
    parser.add_argument("--num-epochs", default=3, type=int)
    parser.add_argument("--warmup-ratio", default=0.05, type=float)  # % based warmup
    parser.add_argument("--logging-steps", default=10, type=int)

    # Quantization
    parser.add_argument("--no-4bit", action="store_true")

    args = parser.parse_args()

    # -------------------------------
    # GPU Check
    # -------------------------------
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU: {name} — {mem_gb:.1f} GB")
    else:
        print("❌ No GPU detected — aborting, RunPod always should have GPU.")
        sys.exit(1)

    # -------------------------------
    # Load JSONL
    # -------------------------------
    train_data = load_jsonl(args.train_path)
    val_data = load_jsonl(args.val_path)

    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    # -------------------------------
    # Load model
    # -------------------------------
    use_4bit = (not args.no_4bit)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=use_4bit,
        dtype=None,
    )

    print(f"✓ Loaded model ({'4-bit' if use_4bit else '16-bit'})")

    # -------------------------------
    # LoRA
    # -------------------------------
    print("Detecting LoRA modules...")
    target_modules = detect_lora_modules(model)

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=True,
    )
    print(f"✓ LoRA enabled on {len(target_modules)} modules")

    # -------------------------------
    # Formatting
    # -------------------------------
    print("Applying prompt formatting...")

    fmt_model_name = args.model_name

    train_ds = train_ds.map(
        lambda b: format_batch(b, fmt_model_name),
        batched=True,
        remove_columns=train_ds.column_names
    )

    val_ds = val_ds.map(
        lambda b: format_batch(b, fmt_model_name),
        batched=True,
        remove_columns=val_ds.column_names
    )

    # -------------------------------
    # Trainer
    # -------------------------------
    print("Starting training...")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=500,  # Save checkpoints during training
        eval_strategy="steps",
        eval_steps=500,  # Evaluate during training to catch overfitting
        load_best_model_at_end=True,  # Load best model based on eval loss
        fp16=not use_4bit,
        report_to="none",  # Disable wandb/tensorboard
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",  # CRITICAL: tells trainer which field contains the formatted text
        max_seq_length=args.max_seq_length,  # CRITICAL: set sequence length
        packing=False,  # Don't pack sequences - ensures proper loss masking
        args=training_args,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    print("\n✓ Training complete!")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
