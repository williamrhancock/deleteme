"""Generate Task 3: Free-text augmentation using Claude/Grok API."""

import json
import yaml
import os
import re
from pathlib import Path
from tqdm import tqdm
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def call_claude_api(prompt, api_key=None):
    """Call Claude API to generate variations."""
    try:
        import anthropic  # type: ignore  # Optional dependency
        
        if not api_key:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        return response.content[0].text
        
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")
    except Exception as e:
        raise Exception(f"Claude API error: {e}")


def call_gemini_api(prompt, api_key=None):
    """Call Google Gemini API directly to generate variations."""
    try:
        import google.generativeai as genai
        
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Get model - available models: gemini-1.5-flash, gemini-1.5-pro, gemini-pro
        # Note: gemini-2.0-flash-exp may not be available via direct API yet
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        
        # List available models to find the right one
        try:
            available_models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    # Extract just the model name (remove 'models/' prefix if present)
                    model_id = m.name.split('/')[-1]
                    available_models.append(model_id)
            
            # Check if requested model is available
            if model_name not in available_models:
                # Try alternatives in order of preference
                for alt in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]:
                    if alt in available_models:
                        model_name = alt
                        print(f"  Using available model: {model_name}")
                        break
                else:
                    # Use first available if none match
                    if available_models:
                        model_name = available_models[0]
                        print(f"  Using first available model: {model_name}")
        except Exception as list_error:
            # If listing fails, use default
            print(f"  Note: Could not list models, using default: {model_name}")
        
        # Try to create model and generate - if model not found, try alternatives
        alternative_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
        
        for attempt_model in [model_name] + [m for m in alternative_models if m != model_name]:
            try:
                model = genai.GenerativeModel(attempt_model)
                
                # Generate response with retry logic for rate limits
                max_retries = 5
                for retry in range(max_retries):
                    try:
                        response = model.generate_content(
                            prompt,
                            generation_config=genai.types.GenerationConfig(
                                temperature=0.9,
                                max_output_tokens=2000,
                            )
                        )
                        
                        if attempt_model != model_name:
                            print(f"  Note: Using {attempt_model} instead of {model_name}")
                        
                        return response.text
                        
                    except Exception as api_error:
                        error_str = str(api_error).lower()
                        # Check for rate limiting - Google uses various error messages
                        if ("429" in str(api_error) or "rate limit" in error_str or 
                            "quota" in error_str or "resource exhausted" in error_str or
                            "too many requests" in error_str):
                            if retry < max_retries - 1:
                                wait_time = (2 ** retry) * 15  # Exponential backoff: 15s, 30s, 60s, 120s, 240s
                                print(f"  ⚠️  Rate limited, waiting {wait_time}s before retry {retry + 1}/{max_retries}...")
                                time.sleep(wait_time)
                                continue
                            else:
                                raise Exception(f"Rate limited after {max_retries} retries. Please wait and try again later.")
                        else:
                            # Not a rate limit error, re-raise
                            raise
                
            except Exception as model_error:
                if "404" in str(model_error) or "not found" in str(model_error).lower():
                    # Try next model
                    continue
                else:
                    # Different error, re-raise
                    raise
        
    except ImportError:
        raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
    except Exception as e:
        raise Exception(f"Google Gemini API error: {e}")


def call_grok_api(prompt, api_key=None):
    """Call API via OpenRouter (Gemini, Grok, etc.) to generate variations."""
    try:
        import requests
        
        if not api_key:
            api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")
        
        # Use OpenRouter REST API directly
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/bruhvector",  # Optional: for tracking
            "X-Title": "BruhVector Gen-Z Translator"  # Optional: for tracking
        }
        # Allow model name to be overridden via environment variable
        # Default to Grok 4.1 Fast free (x-ai/grok-4.1-fast:free)
        # Other options: google/gemini-2.0-flash-exp:free, x-ai/grok-4.1-fast:free
        model_name = os.getenv("MODEL_NAME", os.getenv("GROK_MODEL", "x-ai/grok-4.1-fast:free"))
        
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000,
            "temperature": 0.9  # For variety in variations
        }
        
        # Add timeout to prevent hanging
        response = requests.post(url, json=data, headers=headers, timeout=60)
        
        # Handle rate limiting (429) - retry with backoff
        if response.status_code == 429:
            # Rate limited - wait and retry
            import time
            wait_time = 5  # Start with 5 seconds
            max_retries = 3
            
            for retry in range(max_retries):
                time.sleep(wait_time)
                response = requests.post(url, json=data, headers=headers, timeout=60)
                if response.status_code != 429:
                    break
                wait_time *= 2  # Exponential backoff
                if retry == max_retries - 1:
                    raise Exception(
                        f"Rate limited after {max_retries} retries. "
                        f"Consider using a paid model or increasing delays between requests."
                    )
        
        # Better error handling with full error details
        if response.status_code != 200:
            error_detail = response.text
            try:
                error_json = response.json()
                if "error" in error_json:
                    error_obj = error_json["error"]
                    error_detail = error_obj.get("message", str(error_obj))
                    # Include error type if available
                    if "type" in error_obj:
                        error_detail = f"{error_obj['type']}: {error_detail}"
                else:
                    error_detail = str(error_json)
            except:
                pass
            raise Exception(
                f"OpenRouter API error ({response.status_code}): {error_detail}\n"
                f"Model: {data['model']}\n"
                f"Prompt preview: {prompt[:150]}..."
            )
        
        result = response.json()
        
        # Check if response has expected structure
        if "choices" not in result or len(result["choices"]) == 0:
            raise Exception(f"Unexpected API response format: {result}")
        
        return result["choices"][0]["message"]["content"]
        
    except ImportError:
        raise ImportError("requests package not installed")
    except requests.exceptions.RequestException as e:
        raise Exception(f"OpenRouter/Grok API request error: {e}")
    except Exception as e:
        raise Exception(f"OpenRouter/Grok API error: {e}")


def parse_variations(response_text):
    """Parse variations from API response."""
    variations = []
    
    # Try to extract numbered list items
    lines = response_text.split('\n')
    for line in lines:
        line = line.strip()
        # Remove numbering (1., 2., etc.)
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        # Remove bullet points
        line = re.sub(r'^[-•*]\s*', '', line)
        # Remove quotes
        line = line.strip('"\'')
        
        if line and len(line) > 5:  # Minimum length check
            variations.append(line)
    
    return variations[:20]  # Limit to 20


def generate_task3():
    """Generate Task 3 training data: Free-text augmentation."""
    config = load_config()
    
    # Paths
    processed_dir = Path(config['paths']['data_processed'])
    training_dir = Path(config['paths']['data_training'])
    training_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = processed_dir / "examples_with_metadata.jsonl"
    output_path = training_dir / "task3_augmented_variations.jsonl"
    
    print(f"Loading examples from {input_path}...")
    
    # Load examples
    examples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} examples")
    
    # Check for API key - default to OpenRouter with Grok
    api_provider = os.getenv("API_PROVIDER", "openrouter").lower()
    
    if api_provider == "claude":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        model_name = None
    elif api_provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
            print(f"\nUsing Google Gemini API directly")
            print(f"Model: {model_name}")
        else:
            # Fall back to OpenRouter if no direct Google key
            api_provider = "openrouter"
            api_key = os.getenv("OPENROUTER_API_KEY")
            model_name = os.getenv("MODEL_NAME", os.getenv("GROK_MODEL", "x-ai/grok-4.1-fast:free"))
            print(f"\nUsing OpenRouter")
            print(f"Model: {model_name}")
    else:
        api_key = os.getenv("OPENROUTER_API_KEY")
        model_name = os.getenv("MODEL_NAME", os.getenv("GROK_MODEL", "x-ai/grok-4.1-fast:free"))
        print(f"\nUsing OpenRouter")
        print(f"Model: {model_name}")
    
    if not api_key:
        print("\n⚠️  No API key found!")
        print("Set one of:")
        print("  export OPENROUTER_API_KEY=your_key   # For Grok/Gemini via OpenRouter (default)")
        print("  export GOOGLE_API_KEY=your_key       # For direct Gemini API")
        print("  export ANTHROPIC_API_KEY=your_key    # For Claude")
        print("  export API_PROVIDER=openrouter|gemini|claude  # Choose provider (default: openrouter)")
        print("  export MODEL_NAME=x-ai/grok-4.1-fast:free  # Override model")
        print("\nGenerating template file instead...")
        
        # Generate template without API calls
        generate_template_file(examples, output_path)
        return output_path
    
    # Configuration for parallel processing
    max_workers = int(os.getenv("MAX_WORKERS", "5"))  # Number of parallel requests
    batch_size = int(os.getenv("BATCH_SIZE", "20"))  # Process in batches
    save_interval = int(os.getenv("SAVE_INTERVAL", "50"))  # Save every N examples
    
    print(f"\nConfiguration:")
    print(f"  Parallel workers: {max_workers}")
    print(f"  Batch size: {batch_size}")
    print(f"  Save interval: {save_interval} examples")
    
    # Check for existing progress
    existing_pairs = []
    if output_path.exists():
        print(f"\nFound existing output file. Loading progress...")
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    existing_pairs.append(json.loads(line))
        print(f"  Loaded {len(existing_pairs)} existing pairs")
    
    # Track which examples we've already processed
    processed_indices = set()
    if existing_pairs:
        # Extract source indices from existing pairs
        for pair in existing_pairs:
            source_idx = pair.get('metadata', {}).get('source_index', -1)
            if source_idx >= 0:
                processed_indices.add(source_idx)
        print(f"  Found {len(processed_indices)} already processed examples")
    
    # Filter out already processed examples
    examples_to_process = [
        (idx, ex) for idx, ex in enumerate(examples)
        if idx not in processed_indices
    ]
    
    if not examples_to_process:
        print("\n✓ All examples already processed!")
        return output_path
    
    print(f"\nProcessing {len(examples_to_process)} remaining examples...")
    
    # Thread-safe list and lock for writing
    all_variations = existing_pairs.copy()
    write_lock = Lock()
    
    def process_example(example_data):
        """Process a single example and return training pairs."""
        idx, ex = example_data
        description = ex.get('description', '').strip()
        slang = ex.get('slang', '').strip()
        
        if not description:
            return None, idx, None
        
        # Create prompt
        safe_description = description.replace('"', '\\"') if description else ""
        safe_slang = slang.replace('"', '\\"') if slang else ""
        
        if not safe_description:
            return None, idx, None
        
        prompt = f"""Generate 15 different ways a 17-year-old American would say: "{safe_description}"

Examples must feel natural, include typos, emojis, and current 2024-2025 slang.

Output as a numbered list."""
        
        try:
            # Call API
            if api_provider == "claude":
                response = call_claude_api(prompt, api_key)
            elif api_provider == "gemini":
                response = call_gemini_api(prompt, api_key)
            else:
                # Use OpenRouter for Grok
                if model_name:
                    os.environ["MODEL_NAME"] = model_name
                response = call_grok_api(prompt, api_key)
            
            # Parse variations
            variations = parse_variations(response)
            
            # Create training pairs
            training_pairs = []
            for variation in variations:
                training_pair = {
                    "instruction": f'Input: "{variation}"\n\nOutput: A clean, professional English version of the same sentence.',
                    "output": description,
                    "metadata": {
                        "original_slang": slang,
                        "original_description": description,
                        "variation": variation,
                        "source_index": idx
                    }
                }
                training_pairs.append(training_pair)
            
            return training_pairs, idx, None
            
        except Exception as e:
            error_msg = str(e)
            return None, idx, error_msg
    
    # Process in batches with parallel workers
    total_processed = 0
    total_errors = 0
    
    for batch_start in tqdm(range(0, len(examples_to_process), batch_size), desc="Processing batches"):
        batch = examples_to_process[batch_start:batch_start + batch_size]
        
        # Process batch in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_example = {
                executor.submit(process_example, ex_data): ex_data 
                for ex_data in batch
            }
            
            batch_results = []
            for future in as_completed(future_to_example):
                try:
                    result = future.result()
                    batch_results.append(result)
                except Exception as e:
                    ex_data = future_to_example[future]
                    batch_results.append((None, ex_data[0], str(e)))
        
        # Collect results and save incrementally
        batch_pairs = []
        for training_pairs, idx, error_msg in batch_results:
            if error_msg:
                total_errors += 1
                slang = examples[idx].get('slang', 'unknown')
                print(f"\n⚠️  Error processing example {idx} (slang: {slang}): {error_msg}")
                
                # Handle rate limiting - wait before next batch
                if ("429" in error_msg or "rate limit" in error_msg.lower() or 
                    "quota" in error_msg.lower() or "resource exhausted" in error_msg.lower()):
                    wait_time = 30  # Wait 30 seconds for rate limits
                    print(f"  ⚠️  Rate limited. Waiting {wait_time} seconds before next batch...")
                    time.sleep(wait_time)
            elif training_pairs:
                batch_pairs.extend(training_pairs)
                total_processed += 1
        
        # Thread-safe append
        with write_lock:
            all_variations.extend(batch_pairs)
            
            # Save incrementally
            if len(all_variations) % save_interval < len(batch_pairs) or batch_start + batch_size >= len(examples_to_process):
                with open(output_path, 'w', encoding='utf-8') as f:
                    for pair in all_variations:
                        f.write(json.dumps(pair, ensure_ascii=False) + '\n')
                print(f"\n  💾 Saved progress: {len(all_variations)} total pairs ({total_processed} examples processed, {total_errors} errors)")
        
        # Small delay between batches to avoid overwhelming the API
        if batch_start + batch_size < len(examples_to_process):
            time.sleep(2)  # 2 second delay between batches
    
    print(f"\n✓ Generated {len(all_variations)} total augmented training pairs")
    print(f"  Processed: {total_processed} examples")
    print(f"  Errors: {total_errors} examples")
    print(f"  Saved to: {output_path}")
    
    return output_path


def generate_template_file(examples, output_path):
    """Generate a template file showing the expected format."""
    print("\nGenerating template file (no API calls)...")
    
    template_pairs = []
    
    for ex in examples[:10]:  # Just first 10 as examples
        description = ex.get('description', '').strip()
        slang = ex.get('slang', '').strip()
        
        # Create example variations manually
        example_variations = [
            f"that's {slang.lower()} fr",
            f"so {slang.lower()} ngl",
            f"{slang} vibes",
            f"literally {slang.lower()}",
            f"that's {slang.lower()} asf"
        ]
        
        for variation in example_variations:
            training_pair = {
                "instruction": f'Input: "{variation}"\n\nOutput: A clean, professional English version of the same sentence.',
                "output": description,
                "metadata": {
                    "original_slang": slang,
                    "original_description": description,
                    "variation": variation,
                    "note": "This is a template example. Run with API key for real variations."
                }
            }
            template_pairs.append(training_pair)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in template_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"✓ Generated template file with {len(template_pairs)} example pairs")
    print("  Set API key and run again for full augmentation")


if __name__ == "__main__":
    import re
    generate_task3()

