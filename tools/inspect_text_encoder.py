#!/usr/bin/env python3
"""
Text Encoder Structure Inspector

This script helps you understand the structure of different text encoder models,
so you can make informed decisions about which layers to freeze.

Usage:
    python tools/inspect_text_encoder.py --model "sentence-transformers/paraphrase-MiniLM-L6-v2"
    python tools/inspect_text_encoder.py --model "bert-base-uncased" --verbose
    python tools/inspect_text_encoder.py --list-models
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoModel, AutoTokenizer
from tabulate import tabulate


# Common text encoder models
COMMON_MODELS = {
    'MiniLM-L6': 'sentence-transformers/paraphrase-MiniLM-L6-v2',
    'MiniLM-L12': 'sentence-transformers/paraphrase-MiniLM-L12-v2',
    'MPNet': 'sentence-transformers/all-mpnet-base-v2',
    'BERT-base': 'bert-base-uncased',
    'BERT-large': 'bert-large-uncased',
    'RoBERTa-base': 'roberta-base',
    'DistilBERT': 'distilbert-base-uncased',
    'ALBERT-base': 'albert-base-v2',
}


def count_parameters(module):
    """Count total and trainable parameters in a module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def analyze_model_structure(model_name, verbose=False):
    """Analyze and display the structure of a text encoder model."""
    print(f"\n{'='*80}")
    print(f"📊 Analyzing Model: {model_name}")
    print(f"{'='*80}\n")
    
    try:
        # Load model
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
        model = AutoModel.from_pretrained(model_name, local_files_only=False)
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✅ Model loaded successfully!\n")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # 1. Overall Statistics
    print("📈 Overall Statistics")
    print("-" * 80)
    total_params, trainable_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size (MB): {total_params * 4 / 1024 / 1024:.2f}")
    
    if hasattr(model.config, 'hidden_size'):
        print(f"Hidden Size: {model.config.hidden_size}")
    if hasattr(model.config, 'num_hidden_layers'):
        print(f"Number of Layers: {model.config.num_hidden_layers}")
    if hasattr(model.config, 'num_attention_heads'):
        print(f"Attention Heads: {model.config.num_attention_heads}")
    
    print()
    
    # 2. Layer Structure Overview
    print("🏗️  Layer Structure Overview")
    print("-" * 80)
    
    layer_info = []
    current_module = None
    module_params = 0
    
    for name, param in model.named_parameters():
        # Extract top-level module name
        parts = name.split('.')
        if len(parts) > 0:
            module = parts[0]
            if module != current_module:
                if current_module is not None:
                    layer_info.append([current_module, f"{module_params:,}"])
                current_module = module
                module_params = param.numel()
            else:
                module_params += param.numel()
    
    # Add last module
    if current_module is not None:
        layer_info.append([current_module, f"{module_params:,}"])
    
    print(tabulate(layer_info, headers=['Module', 'Parameters'], tablefmt='grid'))
    print()
    
    # 3. Detailed Transformer Layers
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        print("🔍 Transformer Layers Breakdown")
        print("-" * 80)
        
        layers_data = []
        for i, layer in enumerate(model.encoder.layer):
            layer_total, _ = count_parameters(layer)
            layers_data.append([
                f"Layer {i}",
                f"{layer_total:,}",
                f"{layer_total * 100 / total_params:.1f}%"
            ])
        
        print(tabulate(layers_data, 
                      headers=['Layer', 'Parameters', '% of Total'], 
                      tablefmt='grid'))
        print()
    
    # 4. Freezing Recommendations
    print("💡 Freezing Strategy Recommendations (New unfreeze_last_layers Option)")
    print("-" * 80)
    
    if hasattr(model.config, 'num_hidden_layers'):
        num_layers = model.config.num_hidden_layers
        
        recommendations = [
            ["Small Dataset (<5K)", "unfreeze_last_layers: 1", f"Train last 1/{num_layers} layer + projection (Recommended ⭐)"],
            ["Medium Dataset (5K-50K)", "unfreeze_last_layers: 2", f"Train last 2/{num_layers} layers + projection"],
            ["Large Dataset (>50K)", "unfreeze_last_layers: 3", f"Train last 3/{num_layers} layers + projection"],
            ["Very Large Dataset", "freeze_backbone: false", "Fine-tune all layers"],
            ["Minimal Training", "freeze_backbone: true", "Only train projection layer"],
        ]
        
        print(tabulate(recommendations, 
                      headers=['Dataset Size', 'Config (Most Intuitive)', 'Description'], 
                      tablefmt='grid'))
        
        print("\n💡 Alternative: freeze_layers (need to know total layer count)")
        alt_recommendations = [
            ["Small Dataset", f"freeze_layers: {num_layers - 1}", f"Freeze first {num_layers - 1} layers"],
            ["Medium Dataset", f"freeze_layers: {num_layers // 2}", f"Freeze first {num_layers // 2} layers"],
        ]
        print(tabulate(alt_recommendations, 
                      headers=['Dataset Size', 'Config', 'Description'], 
                      tablefmt='grid'))
    
    print()
    
    # 5. Detailed Parameter List (verbose mode)
    if verbose:
        print("📝 Detailed Parameter List")
        print("-" * 80)
        
        param_list = []
        for name, param in model.named_parameters():
            param_list.append([
                name,
                f"{param.numel():,}",
                str(tuple(param.shape)),
                "✓" if param.requires_grad else "✗"
            ])
        
        print(tabulate(param_list, 
                      headers=['Parameter Name', 'Size', 'Shape', 'Trainable'], 
                      tablefmt='simple'))
        print()
    
    # 6. Example Configuration Usage
    print("📋 Example Configuration Usage")
    print("-" * 80)
    
    examples = []
    
    # unfreeze_last_layers examples (Most intuitive)
    examples.append(["🌟 Recommended", "unfreeze_last_layers: 1", "Train only last layer + projection"])
    examples.append(["🌟 Recommended", "unfreeze_last_layers: 2", "Train last 2 layers + projection"])
    
    # freeze_layers examples
    if hasattr(model.config, 'num_hidden_layers'):
        num_layers = model.config.num_hidden_layers
        examples.append(["Alternative", f"freeze_layers: {num_layers - 1}", f"Same as unfreeze_last_layers: 1 (but need to know {num_layers} layers)"])
    
    # freeze_pattern examples (Advanced)
    examples.append(["Advanced", "freeze_pattern: 'embeddings'", "Freeze only embeddings"])
    examples.append(["Advanced", "freeze_pattern: 'encoder.layer.[0-2]'", "Freeze first 3 layers using regex"])
    examples.append(["Advanced", "freeze_pattern: 'attention'", "Freeze all attention layers"])
    
    # Simple examples
    examples.append(["Simple", "freeze_backbone: true", "Freeze all, only train projection"])
    examples.append(["Simple", "freeze_backbone: false", "Fine-tune everything"])
    
    print(tabulate(examples, headers=['Type', 'Configuration', 'Effect'], tablefmt='grid'))
    print()
    
    # 7. Test a sample input
    print("🧪 Testing Model with Sample Input")
    print("-" * 80)
    
    try:
        sample_text = "This is a test sentence."
        inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        if hasattr(outputs, 'last_hidden_state'):
            print(f"Input: '{sample_text}'")
            print(f"Output Shape: {outputs.last_hidden_state.shape}")
            print(f"Hidden Size: {outputs.last_hidden_state.shape[-1]}")
            print("✅ Model works correctly!")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not test model: {e}")
    
    print()


def list_common_models():
    """List common text encoder models."""
    print("\n" + "="*80)
    print("📚 Common Text Encoder Models")
    print("="*80 + "\n")
    
    data = [[alias, model_name] for alias, model_name in COMMON_MODELS.items()]
    print(tabulate(data, headers=['Alias', 'Model Name'], tablefmt='grid'))
    
    print("\nUsage:")
    print("  python tools/inspect_text_encoder.py --model MiniLM-L6")
    print("  python tools/inspect_text_encoder.py --model \"sentence-transformers/paraphrase-MiniLM-L6-v2\"")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect text encoder model structure for freezing decisions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect MiniLM-L6-v2 (default)
  python tools/inspect_text_encoder.py
  
  # Inspect a specific model
  python tools/inspect_text_encoder.py --model "bert-base-uncased"
  
  # Use model alias
  python tools/inspect_text_encoder.py --model MiniLM-L6
  
  # Verbose mode (show all parameters)
  python tools/inspect_text_encoder.py --model MiniLM-L6 --verbose
  
  # List all common models
  python tools/inspect_text_encoder.py --list-models
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='sentence-transformers/paraphrase-MiniLM-L6-v2',
        help='Model name or alias (default: MiniLM-L6-v2)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed parameter list'
    )
    
    parser.add_argument(
        '--list-models', '-l',
        action='store_true',
        help='List common text encoder models'
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        list_common_models()
        return
    
    # Check if model is an alias
    model_name = COMMON_MODELS.get(args.model, args.model)
    
    # Analyze the model
    analyze_model_structure(model_name, verbose=args.verbose)


if __name__ == '__main__':
    main()

