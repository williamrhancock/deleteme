"""Estimate cost for Task 3 augmentation using Grok via OpenRouter."""

import os

def estimate_grok_cost():
    """Estimate cost for processing all examples with Grok."""
    
    # Dataset stats
    num_examples = 1780  # From your processed data
    
    # Per-request token estimates
    # Input: Prompt asking for 15 variations
    prompt_template = """Generate 15 different ways a 17-year-old American would say: "{description}"

Requirements:
- Use current 2024-2025 slang
- Include natural typos and casual punctuation
- Use emojis occasionally (not every example)
- Make it feel authentic and conversational
- Each variation should be a complete sentence or phrase
- Include the slang term "{slang}" naturally in some variations

Format as a numbered list, one variation per line."""
    
    # Average description length: ~50 words = ~75 tokens
    # Average slang term: ~5 tokens
    # Prompt base: ~200 tokens
    avg_input_tokens = 200 + 75 + 5  # ~280 tokens per request
    
    # Output: 15 variations × ~20 words each = ~300 words = ~450 tokens
    avg_output_tokens = 450
    
    # OpenRouter Grok pricing (as of 2024)
    # Using x-ai/grok-4.1-fast:free (FREE!) as default
    # Alternative: x-ai/grok-4-fast ($0.20/$0.50 per million)
    # Alternative: x-ai/grok-4 ($3/$15 per million)
    
    # Check if using free model (default)
    grok_model = os.getenv("GROK_MODEL", "x-ai/grok-4.1-fast:free")
    use_free_model = grok_model.endswith(":free") or grok_model == ""
    
    if use_free_model:
        input_cost_per_million = 0.00  # FREE!
        output_cost_per_million = 0.00  # FREE!
        print(f"Using FREE model: {grok_model if grok_model else 'x-ai/grok-4.1-fast:free'}")
    else:
        # Default to grok-4 pricing if not free
        input_cost_per_million = 3.00
        output_cost_per_million = 15.00
        print(f"Using paid model: {grok_model}")
    
    # Calculate totals
    total_input_tokens = num_examples * avg_input_tokens
    total_output_tokens = num_examples * avg_output_tokens
    total_tokens = total_input_tokens + total_output_tokens
    
    # Calculate costs
    input_cost = (total_input_tokens / 1_000_000) * input_cost_per_million
    output_cost = (total_output_tokens / 1_000_000) * output_cost_per_million
    total_cost = input_cost + output_cost
    
    # OpenRouter platform fees (only if using paid model)
    if use_free_model:
        platform_fee = 0.00  # No fees for free model
        total_with_fees = 0.00  # FREE!
    else:
        # 5.5% fee on credit card purchases (minimum $0.80)
        # OR 5% fee if using BYOK (Bring Your Own Key)
        platform_fee_rate = 0.055  # Credit card
        platform_fee = max(total_cost * platform_fee_rate, 0.80)
        total_with_fees = total_cost + platform_fee
    
    print("=" * 60)
    print("Grok via OpenRouter Cost Estimate")
    print("=" * 60)
    print(f"\nDataset: {num_examples:,} examples")
    print(f"\nPer Request:")
    print(f"  Input tokens:  ~{avg_input_tokens:,} tokens")
    print(f"  Output tokens: ~{avg_output_tokens:,} tokens")
    print(f"  Total:         ~{avg_input_tokens + avg_output_tokens:,} tokens")
    
    print(f"\nTotal Usage:")
    print(f"  Input tokens:  {total_input_tokens:,} tokens ({total_input_tokens/1_000_000:.2f}M)")
    print(f"  Output tokens: {total_output_tokens:,} tokens ({total_output_tokens/1_000_000:.2f}M)")
    print(f"  Total tokens:  {total_tokens:,} tokens ({total_tokens/1_000_000:.2f}M)")
    
    print(f"\nCost Breakdown:")
    print(f"  Input cost:  ${input_cost:.4f} (${input_cost_per_million}/M tokens)")
    print(f"  Output cost: ${output_cost:.4f} (${output_cost_per_million}/M tokens)")
    print(f"  Subtotal:    ${total_cost:.4f}")
    print(f"  Platform fee: ${platform_fee:.4f} (5.5% credit card fee, min $0.80)")
    print(f"  TOTAL:       ${total_with_fees:.4f}")
    
    print(f"\nExpected Output:")
    print(f"  ~{num_examples * 15:,} training pairs (15 variations per example)")
    print(f"  ~{num_examples * 15 / 1000:.1f}K total examples")
    
    print(f"\nCost per 1K examples: ${(total_with_fees / (num_examples * 15 / 1000)):.4f}")
    print(f"Cost per variation: ${(total_with_fees / (num_examples * 15)):.6f}")
    
    print("\n" + "=" * 60)
    print("Alternative: Claude 3.5 Sonnet")
    print("=" * 60)
    
    # Claude pricing comparison
    claude_input_per_million = 3.00  # $3/M input
    claude_output_per_million = 15.00  # $15/M output
    
    claude_input_cost = (total_input_tokens / 1_000_000) * claude_input_per_million
    claude_output_cost = (total_output_tokens / 1_000_000) * claude_output_per_million
    claude_total = claude_input_cost + claude_output_cost
    
    print(f"  Claude 3.5 Sonnet: ${claude_total:.4f} (direct, no platform fee)")
    print(f"  Savings: ${total_with_fees - claude_total:.4f}")
    
    print("\n" + "=" * 60)
    print("Recommendations:")
    print("=" * 60)
    print("1. Start with a small batch (10-50 examples) to test")
    print("2. Monitor token usage - actual may vary")
    print("3. Consider rate limiting to avoid API errors")
    print("4. Claude direct API may be cheaper (no platform fee)")
    print("5. Batch processing with retry logic recommended")
    
    return {
        "total_cost": total_with_fees,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "expected_pairs": num_examples * 15
    }


if __name__ == "__main__":
    estimate_grok_cost()

