#!/usr/bin/env python3
"""
Fine-tune Phi-3-mini or Mistral-7B using Unsloth on RunPod CLI.

Usage:
    python runpod_finetune.py --train-path /workspace/train.jsonl --val-path /workspace/val.jsonl --output-dir /workspace/models/finetuned
    
    # Or with config.yaml:
    python runpod_finetune.py --config config.yaml
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

# Check and install dependencies if needed
def check_and_install_dependencies():
    """Check for required packages and install if missing."""
    required_packages = {
        'torch': 'torch',
        'datasets': 'datasets',
        'transformers': 'transformers',
        'unsloth': 'unsloth',
        'trl': 'trl',
    }
    
    missing = []
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        print(f"⚠️  Missing required packages: {', '.join(missing)}")
        print("Installing missing packages...")
        try:
            # Install standard packages
            standard_packages = [pkg for pkg in missing if pkg != 'unsloth']
            if standard_packages:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-q"
                ] + standard_packages)
            
            # Install unsloth separately (from git)
            if 'unsloth' in missing:
                print("Installing Unsloth from GitHub...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-q",
                    "unsloth @ git+https://github.com/unslothai/unsloth.git"
                ])
            
            print("✓ Dependencies installed successfully")
            # Re-import after installation
            for module_name in missing:
                if module_name == 'unsloth':
                    continue  # Will be imported below
                __import__(module_name)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            print("\nPlease install manually:")
            print(f"  pip install {' '.join(missing)}")
            if 'unsloth' in missing:
                print("  pip install 'unsloth @ git+https://github.com/unslothai/unsloth.git'")
            sys.exit(1)

# Check dependencies before importing
check_and_install_dependencies()

# Now import the packages
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments


def load_jsonl(file_path):
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def formatting_prompts_func(examples, model_name):
    """
    Format examples using the official Phi-3 chat template.
    
    CRITICAL: Phi-3 format must NOT have trailing newline after <|end|>
    The model expects: <|user|>\n{instruction}<|end|>\n<|assistant|>\n{output}<|end|>
    """
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []
    
    for instruction, output in zip(instructions, outputs):
        # OFFICIAL PHI-3 CHAT TEMPLATE (no trailing newline after <|end|>)
        if "Phi-3" in model_name:
            text = f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n{output}<|end|>"
        else:
            # Mistral format
            text = f"[INST] {instruction} [/INST] {output} </s>"
        texts.append(text)
    
    return {"text": texts}


def detect_base_dir():
    """Auto-detect base directory (RunPod /workspace, Colab /content, or current dir)."""
    if os.path.exists("/workspace"):  # RunPod
        return "/workspace"
    elif os.path.exists("/content"):  # Colab
        return "/content"
    else:  # Default
        return os.getcwd()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Phi-3 or Mistral with Unsloth")
    
    # Path arguments
    parser.add_argument("--train-path", type=str, help="Path to train.jsonl file")
    parser.add_argument("--val-path", type=str, help="Path to val.jsonl file")
    parser.add_argument("--output-dir", type=str, help="Output directory for model")
    parser.add_argument("--base-dir", type=str, help="Base directory (auto-detected if not provided)")
    
    # Model arguments
    parser.add_argument("--model-name", type=str, default="microsoft/Phi-3-mini-4k-instruct",
                       help="Model name (default: microsoft/Phi-3-mini-4k-instruct)")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                       help="Maximum sequence length (default: 2048)")
    
    # LoRA arguments
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA r (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)")
    
    # Training arguments (optimized for Phi-3)
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                       help="Learning rate (default: 5e-5, optimized for Phi-3)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8,
                       help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs (default: 3)")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Warmup steps (default: 50)")
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine",
                       help="LR scheduler type (default: cosine, better for Phi-3)")
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")
    
    # Other arguments
    parser.add_argument("--load-in-4bit", action="store_true", default=None,
                       help="Use 4-bit quantization (auto-detected if not specified)")
    parser.add_argument("--no-4bit", action="store_true",
                       help="Disable 4-bit quantization (use 16-bit)")
    
    args = parser.parse_args()
    
    # Auto-detect base directory if not provided
    if args.base_dir:
        base_dir = args.base_dir
    else:
        base_dir = detect_base_dir()
        print(f"Auto-detected base directory: {base_dir}")
    
    # Set default paths if not provided
    if not args.train_path:
        args.train_path = os.path.join(base_dir, "train.jsonl")
    if not args.val_path:
        args.val_path = os.path.join(base_dir, "val.jsonl")
    if not args.output_dir:
        args.output_dir = os.path.join(base_dir, "models", "finetuned")
    
    # Check GPU
    print("Checking GPU availability...")
    if torch.cuda.is_available():
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  No GPU detected. Training will be slower on CPU.")
        if not args.no_4bit:
            args.no_4bit = True  # Disable 4-bit on CPU
    
    # Determine 4-bit usage
    if args.no_4bit:
        use_4bit = False
    elif args.load_in_4bit:
        use_4bit = True
    else:
        # Auto-detect
        try:
            import bitsandbytes as bnb
            use_4bit = hasattr(bnb, 'cuda_available') and bnb.cuda_available() and torch.cuda.is_available()
        except:
            use_4bit = False
    
    # Load training data
    print(f"\nLoading training data...")
    print(f"  Train: {args.train_path}")
    print(f"  Val: {args.val_path}")
    
    train_data = load_jsonl(args.train_path)
    val_data = load_jsonl(args.val_path)
    
    if len(train_data) == 0:
        raise ValueError(f"No training data found in {args.train_path}!")
    if len(val_data) == 0:
        raise ValueError(f"No validation data found in {args.val_path}!")
    
    print(f"✓ Loaded {len(train_data)} training examples and {len(val_data)} validation examples")
    
    # Load model
    print(f"\nLoading model: {args.model_name}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"4-bit quantization: {use_4bit}")
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            dtype=None,  # Auto detection
            load_in_4bit=use_4bit,
        )
        if use_4bit:
            print("✓ Model loaded with 4-bit quantization")
        else:
            print("✓ Model loaded in 16-bit precision")
    except Exception as e:
        if use_4bit:
            print(f"⚠️  4-bit loading failed: {e}")
            print("Falling back to 16-bit precision...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model_name,
                max_seq_length=args.max_seq_length,
                dtype=None,
                load_in_4bit=False,
            )
            print("✓ Model loaded in 16-bit precision")
        else:
            raise
    
    # Add LoRA adapters
    print("\nAdding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    print("✓ Model loaded with LoRA adapters")
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Apply correct formatting
    print("Applying Phi-3 chat template formatting...")
    train_dataset = train_dataset.map(
        lambda examples: formatting_prompts_func(examples, args.model_name),
        batched=True
    )
    val_dataset = val_dataset.map(
        lambda examples: formatting_prompts_func(examples, args.model_name),
        batched=True
    )
    
    print(f"✓ Created datasets: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Show sample
    if len(train_dataset) > 0:
        sample_text = train_dataset[0]["text"]
        print(f"\nSample formatted text (first 200 chars):")
        print(f"  {sample_text[:200]}...")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    print("\nSetting up training arguments...")
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        logging_steps=args.logging_steps,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=3407,
        output_dir=str(output_path),
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to="none",
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
    )
    
    # Print training config
    print("\n" + "="*60)
    print("Training Configuration (optimized for Phi-3):")
    print("="*60)
    print(f"  Model: {args.model_name}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size} (effective: {args.batch_size * args.gradient_accumulation_steps})")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  LR scheduler: {args.lr_scheduler_type}")
    print(f"  Max sequence length: {args.max_seq_length}")
    print(f"  LoRA r: {args.lora_r}, alpha: {args.lora_alpha}, dropout: {args.lora_dropout}")
    print(f"  Output dir: {output_path}")
    print("="*60)
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save model
    print(f"\nSaving model to {output_path}...")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    # Save merged 16-bit model if using quantization
    if use_4bit:
        FastLanguageModel.for_inference(model)
        merged_path = output_path / "merged_16bit"
        model.save_pretrained_merged(
            str(merged_path),
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"✓ Merged 16-bit model saved to {merged_path}")
    
    print(f"\n✓ Training complete! Model saved to {output_path}")
    return output_path


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

