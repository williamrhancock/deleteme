# BruhVector - Gen-Z/Kid Sentiment Translator

Fine-tune Phi-3-mini or Mistral-7B-Instruct to translate gen-z/kid casual language to formal language while preserving sentiment and detecting gender metadata.

## Overview

This project processes the [genz-slang-dataset](https://huggingface.co/datasets/MLBtrio/genz-slang-dataset) to create training pairs for fine-tuning a language model that can translate gen-z/kid speak to formal language.

## Features

- Processes gen-z slang dataset with examples and context
- Extracts sentiment (positive/negative/neutral) from text
- Detects gender from linguistic patterns
- Generates **three different training tasks**:
  1. **Task 1**: Slang → Structured JSON extraction (intent, tone, sentiment, slang terms)
  2. **Task 2**: Slang → Clean formal English translation
  3. **Task 3**: Free-text augmentation (10-20 synthetic variations per example using Claude/Grok)
- Fine-tunes models using Unsloth with LoRA

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

**Option A: Use the installation script (recommended):**
```bash
./install.sh
```

**Option B: Manual installation:**
```bash
# Install torch first
pip install torch>=2.0.0

# Then install the rest
pip install -r requirements.txt
```

**Note for macOS/CPU users:** 
- The main requirements exclude `unsloth` and `xformers` (which don't build on Mac)
- For fine-tuning, use a service provider like Google Colab, Kaggle, or a cloud GPU instance
- The fine-tuning script (`scripts/fine_tune.py`) is included for use on GPU-enabled environments
- For local data processing (download, process, add metadata), the main requirements are sufficient

3. Download NLTK data (if using):
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')"
```

## Usage

### Quick Start (Run Full Pipeline)

Run the complete pipeline from download to fine-tuning:

```bash
python scripts/run_pipeline.py
```

### Step-by-Step

1. Download the dataset:
```bash
python scripts/download_datasets.py
```

2. Process the dataset and generate formal translations:
```bash
python scripts/process_slang_dataset.py
```

3. Add sentiment and gender metadata:
```bash
python scripts/add_metadata.py
```

4. Format for Unsloth training:
```bash
python scripts/format_for_unsloth.py
```

5. Generate training tasks (three different formats):
```bash
# Generate all three tasks
python scripts/generate_all_tasks.py

# Or generate individually:
python scripts/generate_task1_json.py      # Task 1: Slang → JSON
python scripts/generate_task2_translation.py  # Task 2: Slang → Formal English
python scripts/generate_task3_augmentation.py  # Task 3: Augmentation (needs API key)
```

**Task 3 Note:** Requires API key (Grok 4.1 Fast free via OpenRouter recommended):
```bash
export OPENROUTER_API_KEY=your_key  # For Grok via OpenRouter (default, FREE for grok-4.1-fast:free)
export API_PROVIDER=openrouter      # Default provider
export MODEL_NAME=x-ai/grok-4.1-fast:free  # Default model (FREE tier)
```

**Parallel Processing Options:**
```bash
export MAX_WORKERS=5      # Number of parallel API requests (default: 5)
export BATCH_SIZE=20      # Process examples in batches (default: 20)
export SAVE_INTERVAL=50   # Save progress every N examples (default: 50)
```

**Features:**
- ✅ **Parallel processing** - Process multiple examples simultaneously (5x faster)
- ✅ **Automatic progress saving** - Resume if interrupted
- ✅ **Rate limit handling** - Automatic retries with exponential backoff
- ✅ **Free tier support** - Uses Grok 4.1 Fast free model by default

**Cost Estimate:** Processing all 1,780 examples:
- **Free tier** (grok-4.1-fast:free): $0.00 (FREE!)
- **Paid tier** (grok-4.1-fast): ~$14.31
- Generates ~26,700 training pairs (15 variations per example)

Run `python scripts/estimate_cost.py` for detailed breakdown.

6. Prepare data for Unsloth:
```bash
# Use Task 3 augmented data (recommended - 25K+ examples)
python scripts/prepare_for_unsloth.py --task3-only

# Or combine all tasks
python scripts/prepare_for_unsloth.py --all-tasks

# Or use individual tasks
python scripts/prepare_for_unsloth.py --task1-only
python scripts/prepare_for_unsloth.py --task2-only
```

7. Fine-tune the model (on GPU-enabled environment):

**Option A: Google Colab (Recommended)**
1. Open `unsloth_finetune_colab.ipynb` in Google Colab
2. Upload `train.jsonl` and `val.jsonl` files (or clone the repo)
3. Run all cells

**Option B: Local GPU or Kaggle**
```bash
# On Colab/Kaggle/GPU instance:
python scripts/fine_tune.py
```

**Note:** For Mac/CPU users, fine-tuning should be done on a GPU-enabled service (Google Colab, Kaggle, etc.). The data processing scripts (steps 1-6) work fine on Mac.

## Project Structure

```
BruhVector/
├── data/
│   ├── raw/              # Downloaded datasets
│   ├── processed/        # Processed and filtered data
│   └── training/         # Final training format
├── scripts/
│   ├── download_datasets.py
│   ├── process_slang_dataset.py
│   ├── add_metadata.py
│   ├── format_for_unsloth.py
│   └── fine_tune.py
├── src/
│   ├── sentiment_analyzer.py
│   ├── gender_detector.py
│   └── formal_translator.py
├── requirements.txt
├── config.yaml
└── README.md
```

## Configuration

Edit `config.yaml` to:
- Select model (Phi-3-mini or Mistral-7B)
- Adjust training hyperparameters
- Configure LoRA parameters
- Set data processing options

## Output

- Fine-tuned model in `models/finetuned/`
- Training logs and metrics
- Processed dataset statistics

