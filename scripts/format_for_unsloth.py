"""Format examples as instruction-response pairs for Unsloth fine-tuning."""

import os
import yaml
import json
from pathlib import Path
from tqdm import tqdm
import random


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def format_for_unsloth():
    """Format examples as instruction-response pairs."""
    config = load_config()
    
    # Paths
    processed_dir = Path(config['paths']['data_processed'])
    training_dir = Path(config['paths']['data_training'])
    training_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = processed_dir / "examples_with_metadata.jsonl"
    output_path = training_dir / "training_data.jsonl"
    
    print(f"Loading examples from {input_path}...")
    
    # Load examples
    examples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} examples")
    
    # Format as instruction-response pairs
    training_pairs = []
    
    for ex in tqdm(examples, desc="Formatting training pairs"):
        genz_text = ex.get('genz_example', '').strip()
        formal_text = ex.get('formal_translation', '').strip()
        
        if not genz_text or not formal_text:
            continue
        
        # Create instruction-response pair
        instruction = f"Translate to formal: {genz_text}"
        output = formal_text
        
        training_pair = {
            "instruction": instruction,
            "output": output,
            "sentiment": ex.get('sentiment', 'unknown'),
            "gender": ex.get('gender', 'unknown'),
            "slang_term": ex.get('slang', ''),
            "sentiment_score": ex.get('sentiment_score', 0.0),
            "sentiment_confidence": ex.get('sentiment_confidence', 0.0)
        }
        
        training_pairs.append(training_pair)
    
    print(f"\nCreated {len(training_pairs)} training pairs")
    
    # Shuffle
    random.seed(42)
    random.shuffle(training_pairs)
    
    # Split into train/val
    train_split = config['processing'].get('train_split', 0.9)
    val_split = config['processing'].get('val_split', 0.1)
    
    split_idx = int(len(training_pairs) * train_split)
    train_data = training_pairs[:split_idx]
    val_data = training_pairs[split_idx:]
    
    print(f"\nSplit: {len(train_data)} train, {len(val_data)} validation")
    
    # Save training data
    train_path = training_dir / "train.jsonl"
    val_path = training_dir / "val.jsonl"
    
    print(f"\nSaving training data...")
    with open(train_path, 'w', encoding='utf-8') as f:
        for pair in train_data:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    with open(val_path, 'w', encoding='utf-8') as f:
        for pair in val_data:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"✓ Saved training data:")
    print(f"  Train: {train_path} ({len(train_data)} examples)")
    print(f"  Val: {val_path} ({len(val_data)} examples)")
    
    # Print sample
    if training_pairs:
        print(f"\nSample training pair:")
        sample = training_pairs[0]
        print(f"  Instruction: {sample['instruction']}")
        print(f"  Output: {sample['output']}")
        print(f"  Sentiment: {sample['sentiment']} (score: {sample['sentiment_score']:.2f})")
        print(f"  Gender: {sample['gender']}")
    
    return train_path, val_path


if __name__ == "__main__":
    format_for_unsloth()

