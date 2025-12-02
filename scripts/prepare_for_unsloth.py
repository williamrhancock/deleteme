"""Prepare training data from all tasks for Unsloth fine-tuning."""

import json
import yaml
import random
from pathlib import Path
from tqdm import tqdm


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_training_data(use_tasks=None, combine_all=True):
    """
    Prepare training data for Unsloth.
    
    Args:
        use_tasks: List of task numbers to use (e.g., [1, 2, 3]). If None, uses all.
        combine_all: If True, combines all tasks. If False, uses only specified tasks.
    """
    config = load_config()
    
    # Paths
    training_dir = Path(config['paths']['data_training'])
    training_dir.mkdir(parents=True, exist_ok=True)
    
    # Task files
    task_files = {
        1: training_dir / "task1_json_extraction.jsonl",
        2: training_dir / "task2_formal_translation.jsonl",
        3: training_dir / "task3_augmented_variations.jsonl",
    }
    
    # Determine which tasks to use
    if use_tasks is None:
        use_tasks = [1, 2, 3] if combine_all else []
    
    print(f"Preparing training data from tasks: {use_tasks}")
    
    # Load data from all specified tasks
    all_examples = []
    
    for task_num in use_tasks:
        task_file = task_files.get(task_num)
        if not task_file or not task_file.exists():
            print(f"⚠️  Task {task_num} file not found: {task_file}")
            continue
        
        print(f"\nLoading Task {task_num} from {task_file}...")
        task_examples = []
        
        with open(task_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        ex = json.loads(line)
                        # Ensure it has instruction and output
                        if 'instruction' in ex and 'output' in ex:
                            task_examples.append(ex)
                    except json.JSONDecodeError:
                        continue
        
        print(f"  Loaded {len(task_examples)} examples from Task {task_num}")
        all_examples.extend(task_examples)
    
    if not all_examples:
        raise ValueError("No training examples found! Make sure task files exist.")
    
    print(f"\n✓ Total examples: {len(all_examples)}")
    
    # Shuffle
    random.seed(42)
    random.shuffle(all_examples)
    
    # Split into train/val
    train_split = config['processing'].get('train_split', 0.9)
    split_idx = int(len(all_examples) * train_split)
    train_data = all_examples[:split_idx]
    val_data = all_examples[split_idx:]
    
    print(f"\nSplit: {len(train_data)} train, {len(val_data)} validation")
    
    # Save training data
    train_path = training_dir / "train.jsonl"
    val_path = training_dir / "val.jsonl"
    
    print(f"\nSaving training data...")
    with open(train_path, 'w', encoding='utf-8') as f:
        for pair in tqdm(train_data, desc="Saving train"):
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    with open(val_path, 'w', encoding='utf-8') as f:
        for pair in tqdm(val_data, desc="Saving val"):
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Saved training data:")
    print(f"  Train: {train_path} ({len(train_data)} examples)")
    print(f"  Val: {val_path} ({len(val_data)} examples)")
    
    # Print samples
    if train_data:
        print(f"\nSample training pair:")
        sample = train_data[0]
        print(f"  Instruction: {sample['instruction'][:100]}...")
        print(f"  Output: {sample['output'][:100]}...")
    
    return train_path, val_path


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    use_tasks = None
    combine_all = True
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--task3-only":
            use_tasks = [3]
            combine_all = False
        elif sys.argv[1] == "--task2-only":
            use_tasks = [2]
            combine_all = False
        elif sys.argv[1] == "--task1-only":
            use_tasks = [1]
            combine_all = False
        elif sys.argv[1] == "--all-tasks":
            use_tasks = [1, 2, 3]
            combine_all = True
    
    prepare_training_data(use_tasks=use_tasks, combine_all=combine_all)

