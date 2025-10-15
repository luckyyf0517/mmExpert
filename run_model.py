import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import yaml
import glob
import shutil
import random
import argparse
import logging
import numpy as np

import sys
sys.path.append('.')

import torch
torch.set_float32_matmul_precision('high')
torch.autograd.set_detect_anomaly(True)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", ".*Checkpoint directory*")
warnings.filterwarnings('ignore', '.*find_unused_parameters=True was specified.*')
warnings.filterwarnings('ignore', '.*TypedStorage is deprecated.*')
warnings.filterwarnings('ignore', '.*0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives.*')
warnings.filterwarnings("ignore", ".*The pynvml package is deprecated. Please install nvidia-ml-py instead.*")


# Ignore specific PyTorch Lightning warning
warnings.filterwarnings("ignore", message="It is recommended to use `self.log('valid/loss_all', ..., sync_dist=True)` when logging on epoch level in distributed setting to accumulate the metric across devices.")

from pytorch_lightning.utilities import disable_possible_user_warnings
disable_possible_user_warnings()

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from swanlab.integration.pytorch_lightning import SwanLabLogger

from src.misc.io import load_yaml
from src.misc.tools import instantiate_from_config


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest="config", default=None, required=False)
    parser.add_argument("--resume-checkpoint", default=None, type=str, required=False)
    parser.add_argument("--version", '-v', default=None, type=str, required=False)
    parser.add_argument('--seed', dest="seed", default=88, help="random seed")
    parser.add_argument('--test', dest="test", action="store_true", default=False)
    
    args = parser.parse_args()
    if args.test: 
        assert args.resume_checkpoint is not None
    
    args.rank = int(os.environ.get('RANK'))
    args.world_size = int(os.environ.get('WORLD_SIZE'))
    return args


if __name__ == '__main__': 
    args = parse_args()
    
    cfg = load_yaml(args.config)
    if args.version is None: 
        args.version = args.config.replace('.yaml', '').replace('config/', '')
    
    os.makedirs(os.path.join(cfg.log_dir, args.version), exist_ok=True)
    shutil.copy(args.config, os.path.join(cfg.log_dir, args.version, 'config.yaml'))
    
    data_cfg = cfg.data_cfg
    data_cfg.params.cfg.batch_size = data_cfg.params.cfg.batch_size // args.world_size # for each gpu
    data = instantiate_from_config(data_cfg)
    data.setup('fit')
    
    set_seed(seed=args.seed, n_gpu=args.world_size)
    
    model_cfg = cfg.model_cfg
    model = instantiate_from_config(model_cfg)
    
    logger = SwanLabLogger(name=args.version, project='mmExpert')
    if not args.resume_checkpoint and args.rank == 0: 
        for log_file in glob.glob(os.path.join(cfg.log_dir, args.version, '*.ckpt')): 
            os.remove(log_file) 
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.log_dir, args.version),
        monitor='valid/loss_clip',
        filename='epoch_{epoch:02d}_val_{valid/loss_clip:.4f}',
        save_top_k=10,
        mode='min',
        auto_insert_metric_name=False, 
        save_last=True,
        save_weights_only=False,)
    
    trainer = Trainer(
        accelerator='gpu', 
        devices=args.world_size, 
        strategy=cfg.strategy, 
        logger=logger, 
        log_every_n_steps=1, 
        max_epochs=cfg.model_cfg.params.max_epochs, 
        num_sanity_val_steps=2, # run validation step experimentaly
        reload_dataloaders_every_n_epochs=1, 
        callbacks=[checkpoint_callback], 
        enable_progress_bar=True) 

    if not args.test:
        trainer.fit(model, train_dataloaders=data.train_dataloader(), 
                    val_dataloaders=[data.val_dataloader()], 
                    ckpt_path=args.resume_checkpoint if args.resume_checkpoint else None)
    else: 
        trainer.test(model, datamodule=data, ckpt_path=args.resume_checkpoint if args.resume_checkpoint else None)
