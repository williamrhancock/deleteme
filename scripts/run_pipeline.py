"""Run the complete pipeline from download to fine-tuning."""

import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir.parent))

from download_datasets import download_dataset
from process_slang_dataset import process_dataset
from add_metadata import add_metadata
from format_for_unsloth import format_for_unsloth
from fine_tune import fine_tune


def run_pipeline():
    """Run the complete pipeline."""
    print("=" * 60)
    print("BruhVector - Gen-Z/Kid Sentiment Translator Pipeline")
    print("=" * 60)
    
    try:
        # Step 1: Download dataset
        print("\n[1/5] Downloading dataset...")
        download_dataset()
        
        # Step 2: Process dataset
        print("\n[2/5] Processing dataset...")
        process_dataset()
        
        # Step 3: Add metadata
        print("\n[3/5] Adding sentiment and gender metadata...")
        add_metadata()
        
        # Step 4: Format for Unsloth
        print("\n[4/5] Formatting for Unsloth...")
        format_for_unsloth()
        
        # Step 5: Fine-tune
        print("\n[5/5] Fine-tuning model...")
        fine_tune()
        
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError in pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()

