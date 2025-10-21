#!/usr/bin/env python3
"""
Text Encoder Freezing Strategy Tester

This script loads various text encoder models and tests different freezing strategies
to help you understand exactly which parameters get frozen.

Usage:
    python tools/test_freezing_strategies.py
    python tools/test_freezing_strategies.py --model "bert-base-uncased"
    python tools/test_freezing_strategies.py --strategy unfreeze_last_layers --value 1
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from tabulate import tabulate

# Import TextEncoder
from src.encoders.text_encoder import TextEncoder


# Common text encoder models
COMMON_MODELS = {
    'MiniLM-L6': 'sentence-transformers/paraphrase-MiniLM-L6-v2',
    'MiniLM-L12': 'sentence-transformers/paraphrase-MiniLM-L12-v2',
    'MPNet': 'sentence-transformers/all-mpnet-base-v2',
    'BERT-base': 'bert-base-uncased',
    'DistilBERT': 'distilbert-base-uncased',
}


def count_parameters(module):
    """Count total and trainable parameters in a module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def get_layer_freeze_status(encoder):
    """Get detailed freeze status of each layer."""
    if not hasattr(encoder.backbone, 'encoder') or not hasattr(encoder.backbone.encoder, 'layer'):
        return None
    
    layer_status = []
    for i, layer in enumerate(encoder.backbone.encoder.layer):
        total, trainable = count_parameters(layer)
        is_frozen = trainable == 0
        layer_status.append({
            'layer': i,
            'total': total,
            'trainable': trainable,
            'frozen': is_frozen
        })
    
    return layer_status


def test_strategy(model_name, strategy_name, strategy_params):
    """Test a specific freezing strategy."""
    print(f"\n{'='*80}")
    print(f"🧪 Testing: {strategy_name}")
    print(f"   Model: {model_name}")
    print(f"   Params: {strategy_params}")
    print(f"{'='*80}\n")
    
    try:
        # Create encoder with specified strategy
        encoder = TextEncoder(
            embed_dim=256,
            model_name=model_name,
            max_length=77,
            **strategy_params
        )
        
        print("✅ Encoder created successfully!\n")
        
    except Exception as e:
        print(f"❌ Error creating encoder: {e}\n")
        return
    
    # Get parameter statistics
    backbone_total, backbone_trainable = count_parameters(encoder.backbone)
    encoder_total, encoder_trainable = count_parameters(encoder)
    
    # Display summary
    print("📊 Parameter Statistics")
    print("-" * 80)
    summary_data = [
        ["Backbone Total", f"{backbone_total:,}"],
        ["Backbone Trainable", f"{backbone_trainable:,}"],
        ["Backbone Frozen", f"{backbone_total - backbone_trainable:,}"],
        ["Projection + Others", f"{encoder_total - backbone_total:,}"],
        ["Total Trainable", f"{encoder_trainable:,}"],
        ["Trainable Ratio", f"{100 * encoder_trainable / encoder_total:.1f}%"],
    ]
    print(tabulate(summary_data, headers=['Metric', 'Value'], tablefmt='grid'))
    print()
    
    # Display layer-wise status
    layer_status = get_layer_freeze_status(encoder)
    if layer_status:
        print("🔍 Layer-wise Freeze Status")
        print("-" * 80)
        
        layer_data = []
        for status in layer_status:
            layer_num = status['layer']
            is_frozen = status['frozen']
            symbol = "🔒" if is_frozen else "🔓"
            status_text = "FROZEN" if is_frozen else "TRAINABLE"
            
            layer_data.append([
                f"Layer {layer_num}",
                symbol,
                status_text,
                f"{status['trainable']:,} / {status['total']:,}"
            ])
        
        print(tabulate(layer_data, 
                      headers=['Layer', 'Status', 'Mode', 'Trainable/Total'], 
                      tablefmt='grid'))
        print()
    
    # Display which components are trainable
    print("📦 Component Trainability")
    print("-" * 80)
    
    components = []
    
    # Check embeddings
    if hasattr(encoder.backbone, 'embeddings'):
        emb_total, emb_trainable = count_parameters(encoder.backbone.embeddings)
        components.append([
            "Embeddings",
            "🔓 Trainable" if emb_trainable > 0 else "🔒 Frozen",
            f"{emb_trainable:,} / {emb_total:,}"
        ])
    
    # Check encoder layers (summary)
    if layer_status:
        frozen_layers = sum(1 for s in layer_status if s['frozen'])
        trainable_layers = len(layer_status) - frozen_layers
        components.append([
            f"Transformer Layers ({len(layer_status)} total)",
            f"🔓 {trainable_layers} trainable, 🔒 {frozen_layers} frozen",
            ""
        ])
    
    # Check projection
    proj_total, proj_trainable = count_parameters(encoder.projection)
    components.append([
        "Projection Layer",
        "🔓 Trainable" if proj_trainable > 0 else "🔒 Frozen",
        f"{proj_trainable:,} / {proj_total:,}"
    ])
    
    print(tabulate(components, 
                  headers=['Component', 'Status', 'Parameters'], 
                  tablefmt='grid'))
    print()


def run_all_strategies(model_name):
    """Test all freezing strategies on a model."""
    print(f"\n{'#'*80}")
    print(f"# Testing All Freezing Strategies on: {model_name}")
    print(f"{'#'*80}")
    
    strategies = [
        ("No Freezing (Baseline)", {}),
        ("Freeze All Backbone", {"freeze_backbone": True}),
        ("Unfreeze Last 1 Layer", {"unfreeze_last_layers": 1}),
        ("Unfreeze Last 2 Layers", {"unfreeze_last_layers": 2}),
        ("Freeze First 4 Layers", {"freeze_layers": 4}),
        ("Pattern: Freeze Embeddings", {"freeze_pattern": "embeddings"}),
        ("Pattern: Freeze First 3 Layers", {"freeze_pattern": "encoder.layer.[0-2]"}),
    ]
    
    for strategy_name, strategy_params in strategies:
        test_strategy(model_name, strategy_name, strategy_params)


def test_forward_pass(model_name, strategy_params):
    """Test if the encoder works with forward pass."""
    print(f"\n{'='*80}")
    print(f"🚀 Testing Forward Pass")
    print(f"   Model: {model_name}")
    print(f"   Strategy: {strategy_params}")
    print(f"{'='*80}\n")
    
    try:
        # Create encoder
        encoder = TextEncoder(
            embed_dim=256,
            model_name=model_name,
            max_length=77,
            **strategy_params
        )
        
        # Test with sample text
        sample_texts = [
            "A person walks forward.",
            "The human is running quickly.",
            "Someone jumps in the air."
        ]
        
        print(f"Input: {len(sample_texts)} sentences")
        
        # Create ModalityData
        from src.core.base import ModalityData, ModalityType
        modality_data = ModalityData(
            data=sample_texts,
            modality=ModalityType.TEXT
        )
        
        # Encode
        with torch.no_grad():
            result = encoder.encode(modality_data, return_sequence=False)
        
        print(f"✅ Forward pass successful!")
        print(f"   Output Shape: {result.features.shape}")
        print(f"   Expected: [batch_size={len(sample_texts)}, embed_dim=256]")
        
        # Verify shape
        if result.features.shape == (len(sample_texts), 256):
            print(f"   ✓ Shape is correct!\n")
        else:
            print(f"   ✗ Shape mismatch!\n")
        
    except Exception as e:
        print(f"❌ Error during forward pass: {e}\n")


def compare_strategies(model_name):
    """Compare different strategies side by side."""
    print(f"\n{'='*80}")
    print(f"📊 Strategy Comparison: {model_name}")
    print(f"{'='*80}\n")
    
    strategies = [
        ("No Freeze", {}),
        ("Freeze All", {"freeze_backbone": True}),
        ("Unfreeze Last 1", {"unfreeze_last_layers": 1}),
        ("Unfreeze Last 2", {"unfreeze_last_layers": 2}),
        ("Freeze First 3", {"freeze_layers": 3}),
    ]
    
    comparison_data = []
    
    for strategy_name, strategy_params in strategies:
        try:
            encoder = TextEncoder(
                embed_dim=256,
                model_name=model_name,
                **strategy_params
            )
            
            total, trainable = count_parameters(encoder)
            backbone_total, backbone_trainable = count_parameters(encoder.backbone)
            
            comparison_data.append([
                strategy_name,
                f"{trainable:,}",
                f"{100 * trainable / total:.1f}%",
                f"{backbone_trainable:,}",
                f"{total - trainable:,}"
            ])
            
        except Exception as e:
            comparison_data.append([
                strategy_name,
                "Error",
                "-",
                "-",
                "-"
            ])
    
    print(tabulate(comparison_data, 
                  headers=['Strategy', 'Trainable Params', '% Train', 'Backbone Train', 'Frozen Params'],
                  tablefmt='grid'))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Test text encoder freezing strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all strategies on default model (MiniLM-L6-v2)
  python tools/test_freezing_strategies.py
  
  # Test specific model
  python tools/test_freezing_strategies.py --model "bert-base-uncased"
  
  # Test specific strategy
  python tools/test_freezing_strategies.py --strategy unfreeze_last_layers --value 1
  
  # Run all tests (comprehensive)
  python tools/test_freezing_strategies.py --all
  
  # Compare strategies side by side
  python tools/test_freezing_strategies.py --compare
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='sentence-transformers/paraphrase-MiniLM-L6-v2',
        help='Model name or alias (default: MiniLM-L6-v2)'
    )
    
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        choices=['freeze_backbone', 'freeze_layers', 'unfreeze_last_layers', 'freeze_pattern'],
        help='Specific strategy to test'
    )
    
    parser.add_argument(
        '--value', '-v',
        type=str,
        help='Value for the strategy (e.g., "1" for unfreeze_last_layers, "embeddings" for freeze_pattern)'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Test all strategies'
    )
    
    parser.add_argument(
        '--compare', '-c',
        action='store_true',
        help='Compare strategies side by side'
    )
    
    parser.add_argument(
        '--test-forward',
        action='store_true',
        help='Test forward pass with the strategy'
    )
    
    args = parser.parse_args()
    
    # Check if model is an alias
    model_name = COMMON_MODELS.get(args.model, args.model)
    
    # Run tests based on arguments
    if args.compare:
        compare_strategies(model_name)
    elif args.all:
        run_all_strategies(model_name)
    elif args.strategy and args.value:
        # Test specific strategy
        strategy_params = {}
        
        if args.strategy == 'freeze_backbone':
            strategy_params['freeze_backbone'] = args.value.lower() == 'true'
        elif args.strategy == 'freeze_layers':
            strategy_params['freeze_layers'] = int(args.value)
        elif args.strategy == 'unfreeze_last_layers':
            strategy_params['unfreeze_last_layers'] = int(args.value)
        elif args.strategy == 'freeze_pattern':
            strategy_params['freeze_pattern'] = args.value
        
        test_strategy(model_name, f"{args.strategy}={args.value}", strategy_params)
        
        if args.test_forward:
            test_forward_pass(model_name, strategy_params)
    else:
        # Default: run all strategies
        print("💡 Tip: Use --compare for side-by-side comparison, or --all for detailed tests")
        run_all_strategies(model_name)


if __name__ == '__main__':
    main()

