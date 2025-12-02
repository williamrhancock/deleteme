"""Test OpenRouter connection and find available Grok models."""

import os
import requests
import json


def list_grok_models():
    """List available Grok models on OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("⚠️  OPENROUTER_API_KEY not found in environment")
        print("Set it with: export OPENROUTER_API_KEY=your_key")
        return
    
    # Get available models from OpenRouter
    url = "https://openrouter.ai/api/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        models = response.json().get("data", [])
        
        # Filter for Grok models
        grok_models = [m for m in models if "grok" in m.get("id", "").lower()]
        
        print("=" * 60)
        print("Available Grok Models on OpenRouter:")
        print("=" * 60)
        
        if grok_models:
            for model in grok_models:
                model_id = model.get("id", "unknown")
                name = model.get("name", "unknown")
                pricing = model.get("pricing", {})
                print(f"\nModel ID: {model_id}")
                print(f"  Name: {name}")
                if pricing:
                    print(f"  Pricing: {pricing}")
        else:
            print("\nNo Grok models found. Available models:")
            for model in models[:10]:  # Show first 10
                print(f"  - {model.get('id', 'unknown')}")
        
        print("\n" + "=" * 60)
        
        # Test with the first Grok model
        if grok_models:
            test_model = grok_models[0]["id"]
            print(f"\nTesting connection with: {test_model}")
            test_connection(test_model, api_key)
        
    except Exception as e:
        print(f"Error fetching models: {e}")


def test_connection(model_id, api_key):
    """Test API connection with a specific model."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Say 'test' if you can read this."}],
        "max_tokens": 10
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(f"✓ Connection successful!")
            print(f"  Response: {content}")
            print(f"\nUse this model ID in your script: {model_id}")
        else:
            print(f"✗ Connection failed ({response.status_code})")
            try:
                error = response.json()
                print(f"  Error: {error}")
            except:
                print(f"  Error: {response.text}")
                
    except Exception as e:
        print(f"✗ Connection error: {e}")


if __name__ == "__main__":
    list_grok_models()

