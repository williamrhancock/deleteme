"""Fine-tune Phi-3-mini or Mistral-7B using Unsloth."""

import os
import yaml
import json
from pathlib import Path
from tqdm import tqdm
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_training_data(train_path: Path, val_path: Path):
    """Load training and validation data from JSONL files."""
    train_data = []
    val_data = []
    
    print(f"Loading training data from {train_path}...")
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                train_data.append(json.loads(line))
    
    print(f"Loading validation data from {val_path}...")
    with open(val_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                val_data.append(json.loads(line))
    
    print(f"Loaded {len(train_data)} training examples and {len(val_data)} validation examples")
    
    return train_data, val_data


def formatting_prompts_func(examples, model_name):
    """
    Format examples using the official Phi-3 chat template.
    This function is used with dataset.map() for batched processing.
    
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


def fine_tune():
    """Fine-tune the model using Unsloth."""
    config = load_config()
    
    # Paths
    training_dir = Path(config['paths']['data_training'])
    train_path = training_dir / "train.jsonl"
    val_path = training_dir / "val.jsonl"
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    train_data, val_data = load_training_data(train_path, val_path)
    
    if len(train_data) == 0:
        raise ValueError("No training data found!")
    
    # Model configuration
    model_name = config['model']['name']
    max_seq_length = config['model']['max_seq_length']
    
    print(f"\nLoading model: {model_name}")
    print(f"Max sequence length: {max_seq_length}")
    
    # Check if CUDA is available
    use_4bit = torch.cuda.is_available()
    if not use_4bit:
        print("⚠️  No GPU detected. Using CPU mode (slower, but will work).")
        print("   For faster training, consider using a GPU-enabled environment.")
    
    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto detection
        load_in_4bit=use_4bit,  # 4-bit quantization only on GPU
    )
    
    # Add LoRA adapters
    lora_config = config['lora']
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config['r'],
        target_modules=lora_config['target_modules'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        bias=lora_config['bias'],
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    print("✓ Model loaded with LoRA adapters")
    
    # Prepare datasets
    print("\nPreparing datasets...")
    
    # Create datasets from raw data
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Apply correct formatting using batched map (more efficient)
    print("Applying Phi-3 chat template formatting...")
    train_dataset = train_dataset.map(
        lambda examples: formatting_prompts_func(examples, model_name),
        batched=True
    )
    val_dataset = val_dataset.map(
        lambda examples: formatting_prompts_func(examples, model_name),
        batched=True
    )
    
    print(f"✓ Created datasets: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Show sample formatted text for verification
    if len(train_dataset) > 0:
        sample_text = train_dataset[0]["text"]
        print(f"\nSample formatted text (first 200 chars):")
        print(f"  {sample_text[:200]}...")
    
    # Training arguments
    training_config = config['training']
    
    # Adjust batch size for CPU
    batch_size = training_config['batch_size']
    if not torch.cuda.is_available():
        batch_size = min(batch_size, 1)  # Smaller batch size for CPU
        print(f"⚠️  Reducing batch size to {batch_size} for CPU training")
    
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        warmup_steps=training_config['warmup_steps'],
        num_train_epochs=training_config['num_epochs'],
        learning_rate=training_config['learning_rate'],
        fp16=False,  # Disable fp16 on CPU
        bf16=False,   # Disable bf16 on CPU
        logging_steps=training_config['logging_steps'],
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),  # cosine better for Phi-3
        seed=3407,
        output_dir=str(output_dir),
        save_steps=training_config['save_steps'],
        eval_steps=training_config['eval_steps'],
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )
    
    print("\nStarting training...")
    print(f"Training arguments (optimized for Phi-3):")
    print(f"  - Epochs: {training_config['num_epochs']}")
    effective_batch = batch_size * training_config['gradient_accumulation_steps']
    print(f"  - Batch size: {batch_size} (effective: {effective_batch})")
    print(f"  - Gradient accumulation steps: {training_config['gradient_accumulation_steps']}")
    print(f"  - Learning rate: {training_config['learning_rate']}")
    print(f"  - Warmup steps: {training_config['warmup_steps']}")
    print(f"  - LR scheduler: {training_config.get('lr_scheduler_type', 'cosine')}")
    print(f"  - Output dir: {output_dir}")
    
    # Train
    trainer.train()
    
    # Save model
    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Save in 16-bit for inference (only if using quantization)
    if use_4bit:
        FastLanguageModel.for_inference(model)
        model.save_pretrained_merged(
            str(output_dir / "merged_16bit"),
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"✓ Merged 16-bit model saved to {output_dir / 'merged_16bit'}")
    else:
        print(f"✓ Model saved to {output_dir} (CPU mode, no quantization)")
    
    print(f"✓ Model saved to {output_dir}")
    
    return output_dir


if __name__ == "__main__":
    try:
        fine_tune()
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        raise

