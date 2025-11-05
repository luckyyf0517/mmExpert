#!/usr/bin/env python3
"""Validate the training configuration file"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_yaml_config

def validate_config(config_path):
    """Validate configuration file"""
    print(f"🔍 Validating config: {config_path}")

    try:
        cfg = load_yaml_config(config_path)
        print("✅ Config loaded successfully")

        # Check required sections
        required_sections = ['model_cfg', 'data_cfg', 'training']
        for section in required_sections:
            if section not in cfg:
                print(f"❌ Missing required section: {section}")
                return False

        print("✅ All required sections present")

        # Validate model config
        model_cfg = cfg.model_cfg
        required_model_keys = ['model_type', 'encoder_path', 'model']
        for key in required_model_keys:
            if key not in model_cfg:
                print(f"❌ Missing required model config key: {key}")
                return False

        print("✅ Model config is valid")

        # Validate data config
        data_cfg = cfg.data_cfg
        required_data_keys = ['data_root', 'batch_size']
        for key in required_data_keys:
            if key not in data_cfg:
                print(f"❌ Missing required data config key: {key}")
                return False

        print("✅ Data config is valid")

        # Validate training config
        training_cfg = cfg.training
        required_training_keys = ['learning_rate', 'max_epochs']
        for key in required_training_keys:
            if key not in training_cfg:
                print(f"❌ Missing required training config key: {key}")
                return False

        print("✅ Training config is valid")

        # Check if paths exist
        if not os.path.exists(model_cfg.get('encoder_path', '')):
            print(f"⚠️  Encoder path does not exist: {model_cfg.get('encoder_path')}")

        if not os.path.exists(data_cfg.get('data_root', '')):
            print(f"⚠️  Data root does not exist: {data_cfg.get('data_root')}")

        print("✅ Configuration validation completed successfully")
        return True

    except Exception as e:
        print(f"❌ Error validating config: {e}")
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate training configuration")
    parser.add_argument('--config', type=str,
                       default='config/llm/train_llm.yaml',
                       help='Path to configuration file')

    args = parser.parse_args()

    if validate_config(args.config):
        print("\n🎉 Configuration is ready for training!")
        sys.exit(0)
    else:
        print("\n💥 Configuration validation failed!")
        sys.exit(1)