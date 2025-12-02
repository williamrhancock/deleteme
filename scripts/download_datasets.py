"""Download genz-slang-dataset from HuggingFace."""

import os
import yaml
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_dataset():
    """Download the genz-slang-dataset CSV file."""
    config = load_config()
    
    # Create directories
    raw_dir = Path(config['paths']['data_raw'])
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_source = config['dataset']['source']
    dataset_file = config['dataset']['file']
    output_path = raw_dir / dataset_file
    
    print(f"Downloading {dataset_source}/{dataset_file}...")
    
    try:
        # Download from HuggingFace
        downloaded_path = hf_hub_download(
            repo_id=dataset_source,
            filename=dataset_file,
            repo_type="dataset",
            local_dir=str(raw_dir),
            local_dir_use_symlinks=False
        )
        
        # If downloaded to a subdirectory, move to raw_dir
        if downloaded_path != str(output_path):
            import shutil
            shutil.move(downloaded_path, output_path)
        
        print(f"✓ Dataset downloaded to {output_path}")
        
        # Load and inspect
        df = pd.read_csv(output_path)
        print(f"\nDataset loaded: {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nDataset statistics:")
        print(f"  - Total slang terms: {len(df)}")
        print(f"  - Columns: {', '.join(df.columns)}")
        
        return output_path
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print(f"Attempting alternative download method...")
        
        # Alternative: try direct URL
        try:
            import requests
            url = f"https://huggingface.co/datasets/{dataset_source}/resolve/main/{dataset_file}"
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✓ Dataset downloaded to {output_path}")
            return output_path
        except Exception as e2:
            print(f"Error with alternative method: {e2}")
            raise


if __name__ == "__main__":
    download_dataset()

