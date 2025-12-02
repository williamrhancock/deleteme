"""Generate Task 2: Slang → Clean formal English translation."""

import json
import yaml
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


def improve_formal_translation(genz_example, slang, description, context, current_translation):
    """Improve the formal translation to be more natural."""
    import re
    
    # Start with the gen-z example
    formal = genz_example
    
    # Clean up the description to use as replacement
    clean_desc = description
    
    # Remove common prefixes
    clean_desc = re.sub(r'^(Shorthand for|shorthand for|Another way of saying|another way of saying)\s+', '', clean_desc, flags=re.IGNORECASE)
    clean_desc = clean_desc.strip()
    
    # If description is a phrase, extract the key word
    if ',' in clean_desc:
        clean_desc = clean_desc.split(',')[0].strip()
    if '.' in clean_desc and len(clean_desc) > 50:
        clean_desc = clean_desc.split('.')[0].strip()
    
    # Handle specific slang replacements more naturally
    slang_replacements = {
        'W': 'win',
        'L': 'loss',
        'fr': 'for real',
        'no cap': 'no lie',
        'ngl': 'not gonna lie',
        'tbh': 'to be honest',
        'bet': 'sounds good',
        'periodt': '',
        'sheesh': 'wow',
        'bussin': 'delicious',
        'snatched': 'flawless',
        'drip': 'style',
        'slay': 'excel',
        'vibe': 'atmosphere',
        'lowkey': 'somewhat',
        'highkey': 'very',
        'finna': 'going to',
        'cap': 'lie',
        'simp': 'someone who tries too hard',
    }
    
    # Use slang replacement dictionary if available
    if slang.lower() in slang_replacements:
        replacement = slang_replacements[slang.lower()]
        if replacement:
            # Replace slang term (case-insensitive, word boundaries)
            pattern = re.compile(r'\b' + re.escape(slang) + r'\b', re.IGNORECASE)
            formal = pattern.sub(replacement, formal)
        else:
            # Remove the slang term entirely
            pattern = re.compile(r'\b' + re.escape(slang) + r'\b', re.IGNORECASE)
            formal = pattern.sub('', formal).strip()
    else:
        # Generic replacement with cleaned description
        if clean_desc and len(clean_desc) < 30:
            pattern = re.compile(r'\b' + re.escape(slang) + r'\b', re.IGNORECASE)
            formal = pattern.sub(clean_desc, formal)
    
    # Clean up common gen-z patterns
    formal = re.sub(r'\bfr\b', 'for real', formal, flags=re.IGNORECASE)
    formal = re.sub(r'\bno cap\b', 'no lie', formal, flags=re.IGNORECASE)
    formal = re.sub(r'\bngl\b', 'not gonna lie', formal, flags=re.IGNORECASE)
    formal = re.sub(r'\btbh\b', 'to be honest', formal, flags=re.IGNORECASE)
    
    # Remove excessive punctuation
    formal = re.sub(r'!{2,}', '!', formal)
    formal = re.sub(r'\?{2,}', '?', formal)
    
    # Fix spacing
    formal = re.sub(r'\s+', ' ', formal)
    formal = formal.strip()
    
    # Capitalize first letter
    if formal and formal[0].islower():
        formal = formal[0].upper() + formal[1:]
    
    # Ensure proper punctuation
    if formal and not formal.endswith(('.', '!', '?')):
        formal += '.'
    
    return formal


def generate_task2():
    """Generate Task 2 training data: Slang → Clean formal English."""
    config = load_config()
    
    # Paths
    processed_dir = Path(config['paths']['data_processed'])
    training_dir = Path(config['paths']['data_training'])
    training_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = processed_dir / "examples_with_metadata.jsonl"
    output_path = training_dir / "task2_formal_translation.jsonl"
    
    print(f"Loading examples from {input_path}...")
    
    # Load examples
    examples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} examples")
    
    # Initialize translator
    translator = FormalTranslator()
    
    # Generate training pairs
    training_pairs = []
    
    for ex in tqdm(examples, desc="Generating Task 2 pairs"):
        genz_example = ex.get('genz_example', '').strip()
        slang = ex.get('slang', '').strip()
        description = ex.get('description', '').strip()
        context = ex.get('context', '').strip()
        current_translation = ex.get('formal_translation', '').strip()
        
        if not genz_example:
            continue
        
        # Use FormalTranslator for better translation
        formal_output = translator.translate_example(
            example=genz_example,
            slang_term=slang,
            description=description,
            context=context
        )
        
        # If translation is still too similar to original, use improved version
        if formal_output == genz_example or len(formal_output) < len(genz_example) * 0.5:
            formal_output = improve_formal_translation(
                genz_example, slang, description, context, current_translation
            )
        
        # Create training pair
        instruction = f'Input: "{genz_example}"\n\nOutput: A clean, professional English version of the same sentence.'
        output = formal_output
        
        training_pair = {
            "instruction": instruction,
            "output": output,
            "metadata": {
                "slang": slang,
                "genz_example": genz_example,
                "description": description
            }
        }
        
        training_pairs.append(training_pair)
    
    print(f"\nGenerated {len(training_pairs)} training pairs")
    
    # Save
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in training_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"✓ Saved {len(training_pairs)} Task 2 training pairs")
    
    # Print sample
    if training_pairs:
        print(f"\nSample Task 2 pair:")
        sample = training_pairs[0]
        print(f"  Instruction: {sample['instruction']}")
        print(f"  Output: {sample['output']}")
    
    return output_path


if __name__ == "__main__":
    generate_task2()

