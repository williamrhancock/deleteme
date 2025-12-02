"""Add sentiment and gender metadata to processed examples."""

import os
import yaml
import json
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sentiment_analyzer import SentimentAnalyzer
from src.gender_detector import GenderDetector


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def add_metadata():
    """Add sentiment and gender metadata to processed examples."""
    config = load_config()
    
    # Paths
    processed_dir = Path(config['paths']['data_processed'])
    input_path = processed_dir / "processed_examples.jsonl"
    output_path = processed_dir / "examples_with_metadata.jsonl"
    
    print(f"Loading examples from {input_path}...")
    
    # Load examples
    examples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} examples")
    
    # Initialize analyzers
    sentiment_analyzer = SentimentAnalyzer()
    gender_detector = GenderDetector()
    
    min_confidence = config['processing'].get('min_sentiment_confidence', 0.3)
    
    # Process each example
    processed_examples = []
    skipped = 0
    
    for ex in tqdm(examples, desc="Adding metadata"):
        genz_text = ex.get('genz_example', '')
        
        if not genz_text:
            skipped += 1
            continue
        
        # Analyze sentiment
        sentiment_result = sentiment_analyzer.analyze(genz_text)
        
        # Skip if confidence is too low (optional filter)
        if sentiment_result['confidence'] < min_confidence and sentiment_result['sentiment'] == 'neutral':
            # Still include, but mark as low confidence
            pass
        
        # Detect gender
        gender_result = gender_detector.detect(genz_text)
        
        # Add metadata
        ex['sentiment'] = sentiment_result['sentiment']
        ex['sentiment_score'] = sentiment_result['sentiment_score']
        ex['sentiment_confidence'] = sentiment_result['confidence']
        ex['gender'] = gender_result['gender']
        ex['gender_confidence'] = gender_result['confidence']
        ex['gender_method'] = gender_result['method']
        
        processed_examples.append(ex)
    
    print(f"\nProcessed {len(processed_examples)} examples")
    print(f"Skipped {skipped} examples")
    
    # Statistics
    sentiment_counts = {}
    gender_counts = {}
    for ex in processed_examples:
        sentiment = ex.get('sentiment', 'unknown')
        gender = ex.get('gender', 'unknown')
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        gender_counts[gender] = gender_counts.get(gender, 0) + 1
    
    print(f"\nSentiment distribution:")
    for sentiment, count in sorted(sentiment_counts.items()):
        print(f"  {sentiment}: {count}")
    
    print(f"\nGender distribution:")
    for gender, count in sorted(gender_counts.items()):
        print(f"  {gender}: {count}")
    
    # Save
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in processed_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"✓ Saved {len(processed_examples)} examples with metadata")
    
    return output_path


if __name__ == "__main__":
    add_metadata()

