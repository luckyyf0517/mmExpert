import os
import sys
import json
import torch
import shutil
import argparse
from src.misc.io import load_yaml
from src.misc.tools import instantiate_from_config
from termcolor import colored
from easydict import EasyDict as edict


def extract_weights(config_file='config.yaml', checkpoint_path=None):
    """Extract model weights from checkpoint and save to feature directory."""
    # Load configuration
    cfg = load_yaml(config_file)
    version = cfg.get('version', 'default')
    dataset_root = cfg.get('dataset_root', '/root/autodl-tmp/mmExpert-Gen/dataset')
    
    print(colored(f"Extracting weights for version: {version}", 'cyan'))
    
    # Determine checkpoint path
    if checkpoint_path is None:
        checkpoint_path = os.path.join('log', version, 'last.ckpt')
    
    if not os.path.exists(checkpoint_path):
        print(colored(f"Error: Checkpoint not found at {checkpoint_path}", 'red'))
        sys.exit(1)
    
    # Load model
    print(colored(f"Loading checkpoint from {checkpoint_path}", 'cyan'))
    model_cfg = cfg.model_cfg
    model = instantiate_from_config(model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    print(colored(f"Model loaded at epoch {checkpoint['epoch']}, iteration {checkpoint['global_step']}", 'green'))
    
    # Extract encoder and text encoder
    image_encoder = model.image_encoder
    text_encoder = model.text_encoder
    text_projection = model.text_projection
    
    # Create feature directory
    feature_dir = os.path.join('feature', version.replace('/', '_'))
    os.makedirs(feature_dir, exist_ok=True)
    print(colored(f"Saving to {feature_dir}", 'cyan'))
    
    # Save encoder weights
    torch.save(image_encoder.state_dict(), os.path.join(feature_dir, 'encoder.pth'))
    print(colored("Saved encoder.pth", 'green'))
    
    # Save text encoder and projection
    torch.save({
        'text_encoder': text_encoder.state_dict(),
        'text_projection': text_projection.state_dict()
    }, os.path.join(feature_dir, 'text_encoder.pth'))
    print(colored("Saved text_encoder.pth", 'green'))
    
    # Save complete model for evaluation
    torch.save(model.state_dict(), os.path.join(feature_dir, 'clip_model.pth'))
    print(colored("Saved clip_model.pth", 'green'))
    
    # Save configuration
    cfg_new = edict()
    cfg_new.encoder_cfg = model_cfg.params.encoder_cfg
    cfg_new.text_cfg = model_cfg.params.text_cfg
    cfg_new.dataset_cfg = cfg.data_cfg.params.cfg.opt
    cfg_new.dataset_root = dataset_root
    cfg_new.version = version
    
    with open(os.path.join(feature_dir, 'config.json'), 'w') as f:
        json.dump(cfg_new, f, indent=2)
    print(colored("Saved config.json", 'green'))
    
    # Copy split files with correct paths
    train_split = cfg.data_cfg.params.cfg.train_split[0]
    test_split = cfg.data_cfg.params.cfg.test_split[0]
    
    train_split_full = os.path.join(dataset_root, train_split)
    test_split_full = os.path.join(dataset_root, test_split)
    
    if os.path.exists(train_split_full):
        shutil.copy(train_split_full, os.path.join(feature_dir, 'train.json'))
        print(colored("Copied train.json", 'green'))
    
    if os.path.exists(test_split_full):
        shutil.copy(test_split_full, os.path.join(feature_dir, 'test.json'))
        print(colored("Copied test.json", 'green'))
    
    print(colored(f"\nWeights extraction completed! Saved to {feature_dir}", 'green', attrs=['bold']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract model weights to feature directory')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file (default: log/{version}/last.ckpt)')
    args = parser.parse_args()
    
    extract_weights(args.config, args.checkpoint)
