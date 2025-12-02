"""Generate all three training tasks from processed data."""

import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir.parent))

from generate_task1_json import generate_task1
from generate_task2_translation import generate_task2
from generate_task3_augmentation import generate_task3


def generate_all():
    """Generate all three training tasks."""
    print("=" * 60)
    print("Generating All Training Tasks")
    print("=" * 60)
    
    try:
        # Task 1: JSON Extraction
        print("\n[1/3] Generating Task 1: Slang → Structured JSON...")
        task1_path = generate_task1()
        print(f"✓ Task 1 complete: {task1_path}")
        
        # Task 2: Formal Translation
        print("\n[2/3] Generating Task 2: Slang → Clean Formal English...")
        task2_path = generate_task2()
        print(f"✓ Task 2 complete: {task2_path}")
        
        # Task 3: Augmentation (may require API key)
        print("\n[3/3] Generating Task 3: Free-text Augmentation...")
        print("Note: This requires Claude or Grok API key")
        print("Set: export ANTHROPIC_API_KEY=your_key (for Claude)")
        print("  or: export GROK_API_KEY=your_key (for Grok)")
        task3_path = generate_task3()
        print(f"✓ Task 3 complete: {task3_path}")
        
        print("\n" + "=" * 60)
        print("All tasks generated successfully!")
        print("=" * 60)
        print(f"\nOutput files:")
        print(f"  1. {task1_path}")
        print(f"  2. {task2_path}")
        print(f"  3. {task3_path}")
        
    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    generate_all()

