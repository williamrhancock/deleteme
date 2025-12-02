"""Process genz-slang-dataset CSV to extract examples and generate training pairs."""

import os
import yaml
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.formal_translator import FormalTranslator


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def process_dataset():
    """Process the slang dataset and extract examples."""
    config = load_config()
    
    # Paths
    raw_dir = Path(config['paths']['data_raw'])
    processed_dir = Path(config['paths']['data_processed'])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_file = config['dataset']['file']
    input_path = raw_dir / dataset_file
    output_path = processed_dir / "processed_examples.jsonl"
    
    print(f"Loading dataset from {input_path}...")
    
    # Load CSV
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} slang terms")
    
    # Initialize translator
    translator = FormalTranslator()
    
    # Process each row
    examples = []
    skipped = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing examples"):
        slang = str(row.get('Slang', '')).strip()
        description = str(row.get('Description', '')).strip()
        example = str(row.get('Example', '')).strip()
        context = str(row.get('Context', '')).strip()
        
        # Skip if missing essential fields
        if not slang or not example:
            skipped += 1
            continue
        
        # Generate formal translation
        try:
            formal_translation = translator.translate_example(
                example=example,
                slang_term=slang,
                description=description,
                context=context
            )
            
            if not formal_translation or formal_translation == example:
                # If translation didn't change much, try manual replacement
                formal_translation = example.replace(slang, description)
            
            # Create example entry
            example_entry = {
                "slang": slang,
                "description": description,
                "genz_example": example,
                "formal_translation": formal_translation,
                "context": context,
                "source_index": int(idx)
            }
            
            examples.append(example_entry)
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            skipped += 1
            continue
    
    print(f"\nProcessed {len(examples)} examples")
    print(f"Skipped {skipped} examples")
    
    # Save as JSONL
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"✓ Saved {len(examples)} processed examples")
    
    # Print sample
    if examples:
        print(f"\nSample example:")
        sample = examples[0]
        print(f"  Slang: {sample['slang']}")
        print(f"  Gen-Z: {sample['genz_example']}")
        print(f"  Formal: {sample['formal_translation']}")
    
    return output_path


if __name__ == "__main__":
    process_dataset()

