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
import yaml
import glob

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


def load_config_file(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ Error loading config {config_path}: {e}")
        return None


def extract_text_encoder_params(config):
    """Extract text encoder parameters from loaded config."""
    if not config or 'params' not in config or 'encoder_configs' not in config['params']:
        return None, None, None

    encoder_configs = config['params']['encoder_configs']
    if 'text' not in encoder_configs:
        return None, None, None

    text_config = encoder_configs['text']

    # Extract model name
    model_name = text_config.get('model_name', 'sentence-transformers/paraphrase-MiniLM-L6-v2')

    # Extract encoder parameters (excluding common ones handled by TextEncoder)
    encoder_params = {}
    for key, value in text_config.items():
        if key not in ['model_name', 'embed_dim']:  # embed_dim is handled separately
            encoder_params[key] = value

    # Use default embed_dim if not specified
    embed_dim = text_config.get('embed_dim', 256)

    return model_name, embed_dim, encoder_params


def find_config_files(directory, pattern="*.yaml"):
    """Find all YAML config files in a directory."""
    config_files = glob.glob(os.path.join(directory, pattern))
    return sorted(config_files)


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


def test_strategy(model_name, strategy_name, strategy_params, embed_dim=256):
    """Test a specific freezing strategy."""
    print(f"\n{'='*80}")
    print(f"🧪 Testing: {strategy_name}")
    print(f"   Model: {model_name}")
    print(f"   Params: {strategy_params}")
    print(f"{'='*80}\n")

    try:
        # Extract max_length from strategy_params if present, otherwise use default
        max_length = strategy_params.pop('max_length', 77)

        # Create encoder with specified strategy
        encoder = TextEncoder(
            embed_dim=embed_dim,
            model_name=model_name,
            max_length=max_length,
            **strategy_params
        )

        print("✅ Encoder created successfully!\n")
        return encoder

    except Exception as e:
        print(f"❌ Error creating encoder: {e}\n")
        return None


def test_config_file(config_path):
    """Test a configuration file by loading it and creating the text encoder."""
    config_name = os.path.basename(config_path)
    print(f"\n{'='*80}")
    print(f"📁 Testing Config: {config_name}")
    print(f"   Path: {config_path}")
    print(f"{'='*80}\n")

    # Load config
    config = load_config_file(config_path)
    if config is None:
        return False

    # Extract text encoder parameters
    result = extract_text_encoder_params(config)
    if result is None:
        print("❌ Could not extract text encoder parameters from config\n")
        return False

    model_name, embed_dim, encoder_params = result
    if model_name is None:
        print("❌ No model name found in config\n")
        return False

    print(f"📋 Extracted Parameters:")
    print(f"   Model: {model_name}")
    print(f"   Embed Dim: {embed_dim}")
    print(f"   Encoder Params: {encoder_params}\n")

    # Test encoder creation
    encoder = test_strategy(model_name, config_name, encoder_params, embed_dim)

    if encoder is not None:
        # Test forward pass if encoder was created successfully
        test_forward_pass(model_name, encoder_params, embed_dim)
        return True

    return False


def test_config_directory(config_dir):
    """Test all configuration files in a directory."""
    print(f"\n{'#'*80}")
    print(f"# Testing All Configurations in: {config_dir}")
    print(f"{'#'*80}")

    config_files = find_config_files(config_dir)
    if not config_files:
        print(f"❌ No YAML config files found in {config_dir}")
        return

    print(f"📁 Found {len(config_files)} configuration files\n")

    success_count = 0
    total_count = len(config_files)

    for config_path in config_files:
        if test_config_file(config_path):
            success_count += 1

    print(f"\n{'='*80}")
    print(f"📊 Summary: {success_count}/{total_count} configs tested successfully")
    print(f"{'='*80}\n")


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


def test_forward_pass(model_name, strategy_params, embed_dim=256):
    """Test if the encoder works with forward pass."""
    print(f"\n{'='*80}")
    print(f"🚀 Testing Forward Pass")
    print(f"   Model: {model_name}")
    print(f"   Strategy: {strategy_params}")
    print(f"{'='*80}\n")

    try:
        # Make a copy of strategy_params to avoid modifying the original
        params_copy = strategy_params.copy()

        # Extract max_length from strategy_params if present, otherwise use default
        max_length = params_copy.pop('max_length', 77)

        # Create encoder
        encoder = TextEncoder(
            embed_dim=embed_dim,
            model_name=model_name,
            max_length=max_length,
            **params_copy
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
        print(f"   Expected: [batch_size={len(sample_texts)}, embed_dim={embed_dim}]")

        # Verify shape
        if result.features.shape == (len(sample_texts), embed_dim):
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
  python tools/preview_freezing_strategies.py

  # Test specific model
  python tools/preview_freezing_strategies.py --model "bert-base-uncased"

  # Test specific strategy
  python tools/preview_freezing_strategies.py --strategy unfreeze_last_layers --value 1

  # Run all tests (comprehensive)
  python tools/preview_freezing_strategies.py --all

  # Compare strategies side by side
  python tools/preview_freezing_strategies.py --compare

  # Test configuration files
  python tools/preview_freezing_strategies.py --config-dir config/model/experiments-freeze-layers

  # Test single configuration file
  python tools/preview_freezing_strategies.py --config config/model/experiments-freeze-layers/clip_minilm-no_freeze.yaml
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

    parser.add_argument(
        '--config', '-cf',
        type=str,
        help='Test a single configuration file'
    )

    parser.add_argument(
        '--config-dir', '-cd',
        type=str,
        help='Test all configuration files in a directory'
    )

    args = parser.parse_args()

    # Handle config file testing
    if args.config:
        if not os.path.exists(args.config):
            print(f"❌ Config file not found: {args.config}")
            return

        test_config_file(args.config)
        return

    if args.config_dir:
        if not os.path.exists(args.config_dir):
            print(f"❌ Config directory not found: {args.config_dir}")
            return

        test_config_directory(args.config_dir)
        return

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

        encoder = test_strategy(model_name, f"{args.strategy}={args.value}", strategy_params)

        if args.test_forward:
            test_forward_pass(model_name, strategy_params)
    else:
        # Default: run all strategies
        print("💡 Tip: Use --compare for side-by-side comparison, or --all for detailed tests")
        print("💡 To test config files, use --config or --config-dir")
        run_all_strategies(model_name)


if __name__ == '__main__':
    main()

