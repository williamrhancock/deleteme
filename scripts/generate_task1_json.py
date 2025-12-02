"""Generate Task 1: Slang → Structured JSON extraction."""

import json
import yaml
from pathlib import Path
from tqdm import tqdm
import re


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_slang_terms(text, slang_term):
    """Extract slang terms from text."""
    slang_terms = []
    text_lower = text.lower()
    slang_lower = slang_term.lower()
    
    # Check if slang term appears in text
    if slang_lower in text_lower:
        slang_terms.append(slang_term)
    
    # Also check for common variations
    if slang_term == "W" and (" w " in text_lower or text_lower.startswith("w ") or text_lower.endswith(" w")):
        slang_terms.append("W")
    if slang_term == "L" and (" l " in text_lower or text_lower.startswith("l ") or text_lower.endswith(" l")):
        slang_terms.append("L")
    
    return slang_terms if slang_terms else [slang_term]


def generate_task1():
    """Generate Task 1 training data: Slang → Structured JSON."""
    config = load_config()
    
    # Paths
    processed_dir = Path(config['paths']['data_processed'])
    training_dir = Path(config['paths']['data_training'])
    training_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = processed_dir / "examples_with_metadata.jsonl"
    output_path = training_dir / "task1_json_extraction.jsonl"
    
    print(f"Loading examples from {input_path}...")
    
    # Load examples
    examples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} examples")
    
    # Generate training pairs
    training_pairs = []
    
    for ex in tqdm(examples, desc="Generating Task 1 pairs"):
        genz_example = ex.get('genz_example', '').strip()
        slang = ex.get('slang', '').strip()
        description = ex.get('description', '').strip()
        sentiment_score = ex.get('sentiment_score', 0.0)
        sentiment = ex.get('sentiment', 'neutral')
        
        if not genz_example or not slang:
            continue
        
        # Extract slang terms
        slang_terms = extract_slang_terms(genz_example, slang)
        
        # Determine tone (gen-z is the default for this dataset)
        tone = "genz"
        
        # Create JSON output
        json_output = {
            "intent": "express_sentiment" if sentiment != "neutral" else "inform",
            "tone": tone,
            "sentiment": round(sentiment_score, 2),
            "contains_slang": True,
            "slang_terms_detected": slang_terms,
            "normalized_meaning": description
        }
        
        # Create training pair
        instruction = f'User message: "{genz_example}"\n\nOutput only JSON with these keys:\n{{\n  "intent": "...",\n  "tone": "genz|...",\n  "sentiment": -1.0 to 1.0,\n  "contains_slang": true,\n  "slang_terms_detected": ["bussin"],\n  "normalized_meaning": "{{description}}"\n}}'
        output = json.dumps(json_output, ensure_ascii=False)
        
        training_pair = {
            "instruction": instruction,
            "output": output,
            "metadata": {
                "slang": slang,
                "genz_example": genz_example,
                "description": description,
                "sentiment": sentiment,
                "sentiment_score": sentiment_score
            }
        }
        
        training_pairs.append(training_pair)
    
    print(f"\nGenerated {len(training_pairs)} training pairs")
    
    # Save
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in training_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"✓ Saved {len(training_pairs)} Task 1 training pairs")
    
    # Print sample
    if training_pairs:
        print(f"\nSample Task 1 pair:")
        sample = training_pairs[0]
        print(f"  Instruction: {sample['instruction'][:100]}...")
        print(f"  Output: {sample['output']}")
    
    return output_path


if __name__ == "__main__":
    generate_task1()

