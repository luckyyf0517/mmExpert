# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import importlib
from copy import deepcopy
import pytorch_lightning as pl

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset, Sampler

from src.datasets.base_dataset import Text2DopplerDatasetV2


class DInterface(pl.LightningDataModule):
    def __init__(self, cfg):
        super(DInterface, self).__init__()
        self.cfg = cfg
        self.sample_ratio = cfg.sample_ratio
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
    
    def train_dataloader(self):
        dataset = self.train_dataset
        return DataLoader(
            dataset, 
            shuffle=True,
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=True, 
            drop_last=True)

    def val_dataloader(self):
        dataset = self.val_dataset
        return DataLoader(
            dataset, 
            shuffle=True,
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=True, 
            drop_last=True)
    
    def test_dataloader(self):
        dataset = self.test_dataset
        return DataLoader(
            dataset, 
            shuffle=False,
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=True, 
            drop_last=True)


class HumanDInterface(DInterface):
    def __init__(self, cfg):
        super(HumanDInterface, self).__init__(cfg)

    def get_dataset(self, split, scale): 
        opt = deepcopy(self.cfg.opt)
        if isinstance(split, str):
            return Text2DopplerDatasetV2(opt, split, scale)
        elif len(split) == 0: 
            return Text2DopplerDatasetV2(opt, split[0], scale[0])
        else:
            return ConcatDataset([Text2DopplerDatasetV2(opt, split_, scale_) for split_, scale_ in zip(split, scale)])
    
    def setup(self, stage=None):
        if stage == 'fit': 
            self.train_dataset = self.get_dataset(self.cfg.train_split, self.cfg.train_ratio) 
            self.val_dataset = self.get_dataset(self.cfg.val_split, self.cfg.val_ratio) 
            self.test_dataset = self.get_dataset(self.cfg.test_split, self.cfg.test_ratio) 
        elif stage == 'test': 
            self.test_dataset = self.get_dataset(self.cfg.test_split, self.cfg.test_ratio) 
        else: 
            raise NotImplementedError
