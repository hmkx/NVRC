import os
import sys
import shutil
import argparse
import yaml
import copy
import logging
import time
import datetime
import uuid
import re
import fnmatch
from typing import Union, List, Optional, Any, Tuple
from dataclasses import dataclass, field, fields
from functools import partial
from contextlib import nullcontext

import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo as dynamo

import torchvision
import compressai
import deepspeed

import timm
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs

import accelerate

from models.layers import is_network_param, is_grid_param


"""
Tools
"""
def get_ckpt_id():
    return time.strftime('%d-%m-%y_%H%M', time.localtime(time.time())) + '-' + str(uuid.uuid4()).split('-')[0]


def str_to_bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise ValueError


def get_output_path(args):
    if not args.exp_name:
        args.exp_name = '-'.join(
            [args.train_data.dataset] + \
            [args.compress_model['type'], args.model['type'],
             datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
             str(uuid.uuid4()).split('-')[0]]
        )
    output_dir, = accelerate.utils.broadcast_object_list([os.path.join(args.output, args.exp_name)], 0)
    return output_dir


def unwrap_model(model):
    model = model._orig_mod if hasattr(model, '_orig_mod') else model # For compiled models
    model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model # For DDP models
    return model


def extract_parameters(*model_dicts):
    params = []
    for d in model_dicts:
        for m in d.values():
            params += list(m.parameters())
    return params


"""
Accerlator & logger
"""
def get_accelerator_logger(args):
    logger = logging.getLogger(__name__)

    # Accelerator
    accelerator = accelerate.Accelerator(project_config=accelerate.utils.ProjectConfiguration(save_on_each_node=True),
                                         dataloader_config=accelerate.DataLoaderConfiguration(non_blocking=True))

    # Prepare output dir & save configurations
    output_dir = get_output_path(args)
    if accelerator.is_main_process:
        os.makedirs(output_dir)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(yaml.safe_dump(args._to_dict(), default_flow_style=False, indent=4, sort_keys=False))
    accelerator.wait_for_everyone()

    # Log setting
    if accelerator.num_processes > 1:
        formatter = logging.Formatter('[%(asctime)s] ' + 'Rank - ' + \
                                      str(accelerator.process_index) + ' %(levelname)s - %(message)s')
    else:
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')

    os.makedirs(os.path.join(output_dir, f'rank_{accelerator.process_index}'), exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir, f'rank_{accelerator.process_index}', 'logs.txt'))
    file_handler.setFormatter(formatter)
    if accelerator.is_main_process:
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)

    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    # World setting
    logger.info(f'Torch version: {torch.__version__}')
    logger.info(f'World size: {accelerator.num_processes}')
    logger.info(f'Rank: {accelerator.process_index}    Local rank: {accelerator.local_process_index}    Device: {accelerator.device}')
    if accelerator.device.type == 'cuda':
        logger.info(f'Device name: {torch.cuda.get_device_name(accelerator.device)}')

    return accelerator, logger, output_dir


def end_training(args, logger, accelerator):
    accelerator.end_training()


"""
Optimizer & scheduler
"""
def get_optimizer_scheduler(args, logger, accelerator, model_or_params, loader, wd_filter_fn=None):
    args = copy.deepcopy(args)

    # This is to match the accelerator option 'step_scheduler_with_optimizer=True'
    args.sched_on_updates = True 
    updates_per_epoch = len(loader)

    # Convert to number of epochs
    args.decay_milestones = [int(args.epochs * m) for m in args.decay_milestones]

    # Default weight decay filter
    if wd_filter_fn is None:
        wd_filter_fn = lambda k, v: (k.endswith('bias') or v.ndim <= 1)

    # Auto lr scaling - linearly scale the learning rate with the batch size (in frames)
    if args.auto_lr_scaling:
        relative_batch_size = args.train_batch_size * accelerator.num_processes * args.grad_accum * \
                              math.prod(loader.dataset.get_patch_size()[1:]) / \
                              math.prod(loader.dataset.get_video_size()[1:])
        args.lr *= relative_batch_size
        args.warmup_lr *= relative_batch_size
        args.min_lr *= relative_batch_size
        logger.info(f'Autoscale learning rates:')
        logger.info(f'         lr - {args.lr:.2e}')
        logger.info(f'         warmup_lr - {args.warmup_lr:.2e}')
        logger.info(f'         min_lr - {args.min_lr:.2e}')

    # Split the parameters into different groups based on the lr_scale
    args.lr_scale = (args.lr_scale if args.lr_scale is not None else []) + ['default', 1.0]
    assert len(args.lr_scale) % 2 == 0, 'lr_scale should be pairs of key and value.'

    groups = {}
    groups_list = []
    group_keys = args.lr_scale[::2]
    group_lr_scales = args.lr_scale[1::2]
    for gp_k, gp_v in zip(group_keys, group_lr_scales):
        groups[gp_k] = {'params': [], 'keys': [], 'lr_scale': float(gp_v),
                        'weight_decay_orig': float(args.weight_decay), 'weight_decay': float(args.weight_decay)}
        groups[gp_k + '_no_wd'] = {'params': [], 'keys': [], 'lr_scale': float(gp_v),
                                   'weight_decay_orig': 0., 'weight_decay': 0.}

    for k, v in (model_or_params.named_parameters() if isinstance(model_or_params, nn.Module) else model_or_params):
        for gp_k in group_keys:
            if (gp_k.startswith('*') and k.endswith(gp_k[1:])) or (gp_k.endswith('*') and k.startswith(gp_k[:-1])) or \
               (gp_k.startswith('*') and gp_k.endswith('*') and gp_k[1:-1] in k) or gp_k == 'default':
                if not v.requires_grad:
                    continue
                if wd_filter_fn(k, v):
                    groups[gp_k]['keys'].append(k)
                    groups[gp_k]['params'].append(v)
                else:
                    groups[gp_k + '_no_wd']['keys'].append(k)
                    groups[gp_k + '_no_wd']['params'].append(v)
                break

    logger.info(f'Parameter groups:')
    for gp_k in (group_keys[-1:] + group_keys[:-1]):
        gp_v = groups[gp_k]
        gp_v_no_wd = groups[gp_k + '_no_wd']
        groups_list.append(gp_v)
        groups_list.append(gp_v_no_wd)
        logger.info(f'     Key - {gp_k}  Lr scale - {gp_v['lr_scale']}  Weight decay - {gp_v['weight_decay']}  Num of params - {len(gp_v['params'])}')
        logger.info(f'     Key - {gp_k} (No weight decay)  Lr scale - {gp_v_no_wd['lr_scale']}  Weight decay - {gp_v_no_wd['weight_decay']}  Num of params - {len(gp_v_no_wd['params'])}')

    optimizer = create_optimizer_v2(groups_list, **optimizer_kwargs(args))
    scheduler, _ = create_scheduler_v2(optimizer, **scheduler_kwargs(args), updates_per_epoch=updates_per_epoch)
    return optimizer, scheduler


"""
Others
"""
class CheckpointManager:
    def __init__(self, output_dir, name, logger, accelerator, metric_key, save_on_main_process=True):
        self.path = os.path.join(output_dir, name)
        self.logger = logger
        self.accelerator = accelerator
        self.best_metric_key = metric_key
        self.best_metric = -np.inf
        self.save_on_main_process = save_on_main_process

    def _save_meta(self, path, data):
        with open(os.path.join(path, 'meta.yaml'), 'w') as file:
            yaml.dump(data, file)

    def _load_meta(self, path):
        with open(os.path.join(path, 'meta.yaml'), 'r') as file:
            data = yaml.full_load(file)
        return data

    def save(self, epoch, metrics):
        # Save the new checkpoint
        if self.accelerator.is_main_process or not self.save_on_main_process:
            self.accelerator.save_state(output_dir=self.path, safe_serialization=False)
            self._save_meta(self.path, {'epoch': epoch, 'metric': metrics})

    def load(self, path):
        # Load the checkpoint
        self.accelerator.load_state(path)
        return self._load_meta(path)['epoch']


def concate_outputs(src, dst, dataset_name, keep_path=False):
    if os.path.exists(os.path.join(src, dataset_name)):
        os.makedirs(dst, exist_ok=True)
        video_src = os.path.join(src, dataset_name)
        video_dst = os.path.join(dst, dataset_name)
        # Concate the output
        if not os.path.exists(video_dst):
            os.system(f'cp -r {video_src} {video_dst}')
        else:
            os.system(f'cp -r {video_src}/* {video_dst}')
        # Remove the source
        if not keep_path:
            os.system(f'rm -r {video_src}')
    if os.path.exists(os.path.join(src, dataset_name + '.yuv')):
        os.makedirs(dst, exist_ok=True)
        video_src = os.path.join(src, dataset_name + '.yuv')
        video_dst = os.path.join(dst, dataset_name + '.yuv')
        # Concate the output
        if not os.path.exists(video_dst):
            os.system(f'cp {video_src} {video_dst}')
        else:
            os.system(f'cat {video_src} >> {video_dst}')
        # Remove the source
        if not keep_path:
            os.system(f'rm -r {video_src}')