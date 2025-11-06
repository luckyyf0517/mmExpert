import os
import sys
import torch
import argparse
import yaml
from termcolor import colored
from easydict import EasyDict as edict

# Add project root to path
sys.path.append('/root/autodl-tmp/mmExpert')

from src.clip.utils.io import load_config
from src.clip.utils.tools import instantiate_from_config
from src.clip import CLIP


def discover_config_from_checkpoint(checkpoint_path):
    """Auto-discover config file from checkpoint path"""
    model_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    
    # Extract version directory from path
    if os.path.basename(model_dir) == 'checkpoints':
        version_dir = os.path.dirname(model_dir)
    else:
        version_dir = model_dir
    
    # Look for config files in the version directory/config
    config_dir = os.path.join(version_dir, 'config')
    model_config = os.path.join(config_dir, 'model_config.yaml')
    data_config = os.path.join(config_dir, 'data_config.yaml')
    single_config = os.path.join(config_dir, 'config.yaml')
    
    # Check for different config patterns
    if os.path.exists(model_config) and os.path.exists(data_config):
        return merge_configs(model_config, data_config), version_dir
    elif os.path.exists(single_config):
        return single_config, version_dir
    elif os.path.exists(model_config):
        return model_config, version_dir
    
    return None, version_dir

def merge_configs(model_config_path, data_config_path):
    """Merge model and data configs into a single config"""
    import tempfile
    
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    merged_config = {
        'model_cfg': model_config,
        'data_cfg': data_config
    }
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='_merged_config.yaml', delete=False)
    yaml.dump(merged_config, temp_file, default_flow_style=False)
    temp_file.close()
    
    return temp_file.name

def extract_weights(config_file=None, checkpoint_path=None):
    """Extract radar encoder weights from checkpoint and save to feature directory."""
    
    if checkpoint_path is None:
        print(colored("Error: --checkpoint is required", 'red'))
        sys.exit(1)
    
    if not os.path.exists(checkpoint_path):
        print(colored(f"Error: Checkpoint not found at {checkpoint_path}", 'red'))
        sys.exit(1)
    
    # Auto-discover config if not provided
    if config_file is None:
        config_file, version_dir = discover_config_from_checkpoint(checkpoint_path)
        if config_file is None:
            print(colored("Error: Could not auto-discover config file. Please specify --config", 'red'))
            sys.exit(1)
        print(colored(f"[DISCOVER] Auto-discovered config: {config_file}", 'cyan'))
        print(colored(f"[DISCOVER] Version directory: {version_dir}", 'cyan'))
    else:
        model_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        if os.path.basename(model_dir) == 'checkpoints':
            version_dir = os.path.dirname(model_dir)
        else:
            version_dir = model_dir
    
    # Load configuration
    cfg = load_config(config_file)
    version_name = os.path.basename(version_dir)
    dataset_root = cfg.get('dataset_root', '/root/autodl-tmp/mmExpert-Gen/dataset')
    
    print(colored(f"[EXTRACT] Extracting weights for version: {version_name}", 'cyan'))
    
    # Load model
    print(colored(f"[LOAD] Loading checkpoint from {checkpoint_path}", 'cyan'))
    model_cfg = cfg['model_cfg']['params']
    model = CLIP(**model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        global_step = checkpoint.get('global_step', 'unknown')
    else:
        model.load_state_dict(checkpoint)
        epoch = 'unknown'
        global_step = 'unknown'
    
    model.eval()
    
    print(colored(f"[LOAD] Model loaded at epoch {epoch}, iteration {global_step}", 'green'))
    
    # Extract radar encoder (not image_encoder)
    if not hasattr(model, 'radar_encoder'):
        print(colored("Error: Model does not have radar_encoder attribute", 'red'))
        sys.exit(1)
    
    radar_encoder = model.radar_encoder
    
    # Create output directory in feature directory
    output_dir = os.path.join('feature', version_name)
    os.makedirs(output_dir, exist_ok=True)
    print(colored(f"[SAVE] Saving to {output_dir}", 'cyan'))
    
    # Save radar encoder weights
    radar_encoder_path = os.path.join(output_dir, 'radar_encoder.pth')
    torch.save(radar_encoder.state_dict(), radar_encoder_path)
    print(colored(f"[SAVE] Saved radar encoder weights to {radar_encoder_path}", 'green'))
    
    # Save radar encoder configuration (YAML only)
    radar_config = model_cfg['encoder_configs']['radar']
    
    # Convert EasyDict to regular dict recursively to avoid YAML special tags
    def easydict_to_dict(obj):
        """Recursively convert EasyDict to regular dict"""
        if isinstance(obj, edict):
            return {k: easydict_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, dict):
            return {k: easydict_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [easydict_to_dict(item) for item in obj]
        else:
            return obj
    
    # Convert radar_config to regular dict
    radar_config_dict = easydict_to_dict(radar_config)
    
    config_output = {
        'radar_encoder_cfg': radar_config_dict,
        'embed_dim': model_cfg.get('embed_dim', 512),
        'version': version_name,
        'checkpoint_path': checkpoint_path,
        'epoch': epoch,
        'global_step': global_step
    }
    
    config_yaml_path = os.path.join(output_dir, 'radar_encoder_config.yaml')
    with open(config_yaml_path, 'w') as f:
        yaml.dump(config_output, f, default_flow_style=False, allow_unicode=True)
    print(colored(f"[SAVE] Saved radar encoder config to {config_yaml_path}", 'green'))
    
    print(colored(f"\n[SUCCESS] Weights extraction completed! Saved to {output_dir}", 'green', attrs=['bold']))
    print(colored(f"  - Radar encoder weights: {radar_encoder_path}", 'white'))
    print(colored(f"  - Radar encoder config: {config_yaml_path}", 'white'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract radar encoder weights from CLIP checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file (auto-discovered if not specified)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (e.g., log/version/checkpoints/last.ckpt)')
    args = parser.parse_args()
    
    extract_weights(args.config, args.checkpoint)
