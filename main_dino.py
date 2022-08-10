# Copyright (c) Facebook, Inc. and its affiliates.
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
import argparse
import os
import sys
import datetime
import time
import math
import json
import neps
import logging
from pathlib import Path
import pickle

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from eval_linear import eval_linear

import utils
from utils import custom_collate
import vision_transformer as vits
from vision_transformer import DINOHead
from functools import partial
from rpn import RPN
from rpn import ResNetRPN

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # NEPS
    parser.add_argument("--is_neps_run", action="store_true", help="Set this flag to run a NEPS experiment.")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--world_size", default=8, type=int, help="default is for NEPS mode with DDP, so 8.")
    parser.add_argument("--gpu", default=8, type=int, help="default is for NEPS mode with DDP, so 8 GPUs.")
    parser.add_argument('--config_file_path', help="Should be set to a path that does not exist.")
    return parser


def find_free_port():
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def dino_neps_main(working_directory, previous_working_directory, args, **hyperparameters):
    args.output_dir = working_directory
    ngpus_per_node = torch.cuda.device_count()
    print(f"Number of GPUs per node detected: {ngpus_per_node}")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = '29500'
    # os.environ["MASTER_PORT"] = str(find_free_port())
    
    if args.is_neps_run:
        try:
            train_dino(torch.distributed.get_rank(), working_directory, previous_working_directory, args, hyperparameters)
        except:
            return 0

        if torch.distributed.get_rank() == 0: # assumption: rank, running neps is 0
            # Return validation metric
            with open(str(args.output_dir) + "/current_val_metric.txt", "r") as f:
                val_metric = f.read()
            print(f"val_metric: {val_metric}")
            return -float(val_metric)  # Remember: NEPS minimizes the loss!!!
        return 0
    else:
        os.environ["WORLD_SIZE"] = str(args.world_size)
        train_dino(None, args.output_dir, args.output_dir, args)
        

def train_dino(rank, working_directory, previous_working_directory, args, hyperparameters=None):
    if not args.is_neps_run:
        print(f"init distributed mode executed")
        utils.init_distributed_mode(args, rank)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    
    cudnn.benchmark = True
    
    # ============ DINO run with NEPS ============
    if args.is_neps_run:
        args_dict = dict(vars(args))
        for k, v in hyperparameters.items():
            if k in args_dict:
                print(f"{k} : {args_dict[k]} (default: {v}) \n")
            else:
                print(f"{k} : {v}) \n")
                
        print("NEPS hyperparameters: ", hyperparameters)
        
        # Parameterize hyperparameters
        args.lr = hyperparameters["lr"]
        args.out_dim = hyperparameters["out_dim"]
        args.momentum_teacher = hyperparameters["momentum_teacher"]
        args.warmup_teacher_temp = hyperparameters["warmup_teacher_temp"]
        args.warmup_teacher_temp_epochs = hyperparameters["warmup_teacher_temp_epochs"]
        args.weight_decay = hyperparameters["weight_decay"]
        args.weight_decay_end = hyperparameters["weight_decay_end"]
        args.freeze_last_layer = hyperparameters["freeze_last_layer"]
        args.warmup_epochs = hyperparameters["warmup_epochs"]
        args.min_lr = hyperparameters["min_lr"]
        args.drop_path_rate = hyperparameters["drop_path_rate"]
        args.optimizer = hyperparameters["optimizer"]
        args.use_bn_in_head = hyperparameters["use_bn_in_head"]
        args.norm_last_layer = hyperparameters["norm_last_layer"]
        
    else:
        print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    # ============ preparing data ... ============
    # transform = DataAugmentationDINO(
    #     args.global_crops_scale,
    #     args.local_crops_scale,
    #     args.local_crops_number,
    #     args.is_neps_run,
    #     hyperparameters,
    # )
    transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    if args.is_neps_run:
        dataset_percentage_usage = 100
        valid_size = 0.2
        num_train = int(len(dataset) / 100 * dataset_percentage_usage)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        
        np.random.shuffle(indices)

        if np.isclose(valid_size, 0.0):
            train_idx, valid_idx = indices, indices
        else:
            train_idx, valid_idx = indices[split:], indices[:split]
        
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_idx)
    else:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=train_sampler if args.is_neps_run else sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate,
    )
    
    if args.is_neps_run:
        print(f"Data loaded: there are {len(train_idx)} images.")
    else:
        print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    rpn = RPN(backbone=ResNetRPN('resnet18'))
    
    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher, rpn = student.cuda(), teacher.cuda(), rpn.cuda()
    
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        rpn = nn.SyncBatchNorm.convert_sync_batchnorm(rpn)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
        
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    rpn = nn.parallel.DistributedDataParallel(rpn, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        # optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        optimizer = torch.optim.AdamW(params_groups.append([{'params': rpn.parameters()}]))  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
        
    # rpn_optimizer = torch.optim.AdamW(rpn.parameters())
    
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    if previous_working_directory is not None:
        utils.restart_from_checkpoint(
            os.path.join(previous_working_directory, "checkpoint.pth"), 
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            dino_loss=dino_loss,
        )
    else:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, "checkpoint.pth"),  # for DINO baseline
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            dino_loss=dino_loss,
        )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")

    if args.is_neps_run:
        end_epoch = hyperparameters["epoch_fidelity"]
    else:
        end_epoch = args.epochs
   
    try:
        for epoch in range(start_epoch, end_epoch):
            data_loader.sampler.set_epoch(epoch)
            # ============ training one epoch of DINO ... ============
            train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
                data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                epoch, fp16_scaler, rpn, args)
    
            # ============ writing logs ... ============
            save_dict = {
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'args': args,
                'dino_loss': dino_loss.state_dict(),
            }
            if fp16_scaler is not None:
                save_dict['fp16_scaler'] = fp16_scaler.state_dict()
            utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
            if args.saveckp_freq and epoch % args.saveckp_freq == 0:
                utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch}
            if utils.is_main_process():
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
    
        if args.is_neps_run:
            print("\n\n\nStarting Finetuning\n\n\n")
            finetuning_parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
            finetuning_parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
            for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
            finetuning_parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
            help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
            We typically set this to False for ViT-Small and to True with ViT-Base.""")
            finetuning_parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
            finetuning_parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
            finetuning_parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
            finetuning_parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
            finetuning_parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
            finetuning_parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
            training (highest LR used during training). The learning rate is linearly scaled
            with the batch size, and specified here for a reference batch size of 256.
            We recommend tweaking the LR depending on the checkpoint evaluated.""")
            finetuning_parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
            finetuning_parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
            distributed training; see https://pytorch.org/docs/stable/distributed.html""")
            finetuning_parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
            finetuning_parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
            finetuning_parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
            finetuning_parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
            finetuning_parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
            finetuning_parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
            finetuning_parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
            finetuning_parser.add_argument("--is_neps_run", action="store_true", help="Set this flag to run a NEPS experiment.")
            finetuning_parser.add_argument("--do_early_stopping", action="store_true", help="Set this flag to take the best test performance - Default by the DINO implementation.")
            finetuning_parser.add_argument("--world_size", default=8, type=int, help="actually not needed here -- just for avoiding unrecognized arguments error")
            finetuning_parser.add_argument("--gpu", default=8, type=int, help="actually not needed here -- just for avoiding unrecognized arguments error")
            finetuning_parser.add_argument('--config_file_path', help="actually not needed here -- just for avoiding unrecognized arguments error")
            finetuning_args = finetuning_parser.parse_args()
            
            finetuning_args.arch = args.arch
            finetuning_args.data_path = "/data/datasets/ImageNet/imagenet-pytorch/"
            finetuning_args.output_dir = args.output_dir
            finetuning_args.is_neps_run = args.is_neps_run
            finetuning_args.gpu = args.gpu
            finetuning_args.saveckp_freq = 10
            finetuning_args.pretrained_weights = str(finetuning_args.output_dir) + "/checkpoint.pth"
            finetuning_args.seed = args.seed 
            finetuning_args.assert_valid_idx = valid_idx[:10]
            finetuning_args.assert_train_idx = train_idx[:10]
            
            finetuning_args.epochs = 100  # TODO: args.epochs
            finetuning_args.epoch_fidelity = hyperparameters["epoch_fidelity"]
            
            eval_linear(finetuning_args)
            
    except ValueError:
        if args.is_neps_run:
            print("OUTPUT_DIR: ", args.output_dir)
            with open(str(args.output_dir) + "/current_val_metric.txt", "w+") as f:
                f.write(f"{0}\n")


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, rpn, rpn_optimizer, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # print(f"image shape before fw pass: {len(images)} (batch size), {images[0].shape} (shape 1st image), {images[1].shape} (shape 2nd image)")
        
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            images = rpn(images)
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)
            # rpn_loss = -loss

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            raise ValueError("Loss value is invalid.")

        # rpn_optimizer.zero_grad()
    
        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            # rpn.requires_grad_(False)
            # student.requires_grad_(True)
            # teacher.requires_grad_(True)
            
            loss.backward(retain_graph=True)
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
            
            # RPN
            # optimizer.zero_grad()
            # rpn.requires_grad_(True)
            # student.requires_grad_(False)
            # teacher.requires_grad_(False)
    
            # rpn_loss = -dino_loss(student_output, teacher_output, epoch)
            # rpn_loss.backward()
            # rpn_optimizer.step()
        else:
            # rpn.requires_grad_(False)
            # student.requires_grad_(True)
            # teacher.requires_grad_(True)
            
            fp16_scaler.scale(loss).backward(retain_graph=True)
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
            
            # RPN
            # optimizer.zero_grad()
            # rpn.requires_grad_(True)
            # student.requires_grad_(False)
            # teacher.requires_grad_(False)

            # rpn_loss = -dino_loss(student_output, teacher_output, epoch)
    
            # fp16_scaler.scale(rpn_loss).backward()
            # fp16_scaler.step(rpn_optimizer)
            

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, is_neps_run, hyperparameters):
        if is_neps_run:
            p_horizontal_crop_1 = hyperparameters["p_horizontal_crop_1"]
            p_colorjitter_crop_1 = hyperparameters["p_colorjitter_crop_1"]
            p_grayscale_crop_1 = hyperparameters["p_grayscale_crop_1"]
            
            p_horizontal_crop_2 = hyperparameters["p_horizontal_crop_2"]
            p_colorjitter_crop_2 = hyperparameters["p_colorjitter_crop_2"]
            p_grayscale_crop_2 = hyperparameters["p_grayscale_crop_2"]
    
            p_horizontal_crop_3 = hyperparameters["p_horizontal_crop_3"]
            p_colorjitter_crop_3 = hyperparameters["p_colorjitter_crop_3"]
            p_grayscale_crop_3 = hyperparameters["p_grayscale_crop_3"]
        else:
            p_horizontal_crop_1, p_colorjitter_crop_1, p_grayscale_crop_1 = 0.5, 0.8, 0.2
            p_horizontal_crop_2, p_colorjitter_crop_2, p_grayscale_crop_2 = 0.5, 0.8, 0.2
            p_horizontal_crop_3, p_colorjitter_crop_3, p_grayscale_crop_3 = 0.5, 0.8, 0.2
            
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            
            transforms.RandomHorizontalFlip(p=p_horizontal_crop_1),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=p_colorjitter_crop_1
                ),
            transforms.RandomGrayscale(p=p_grayscale_crop_1),
            
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
    
            transforms.RandomHorizontalFlip(p=p_horizontal_crop_2),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=p_colorjitter_crop_2
                ),
            transforms.RandomGrayscale(p=p_grayscale_crop_2),
            
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
    
            transforms.RandomHorizontalFlip(p=p_horizontal_crop_3),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=p_colorjitter_crop_3
                ),
            transforms.RandomGrayscale(p=p_grayscale_crop_3),
            
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # DINO run with NEPS
    if args.is_neps_run:
        utils.init_distributed_mode(args, None)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        )
        pipeline_space = dict(
                    lr=neps.FloatParameter(
                        lower=0.00001, upper=0.01, log=True, default=0.0005, default_confidence="high"
                    ),
                    out_dim=neps.IntegerParameter(
                        lower=1000, upper=100000, log=False, default=65536, default_confidence="high"
                    ),
                    momentum_teacher=neps.FloatParameter(
                        lower=0.8, upper=1, log=True, default=0.996, default_confidence="high"
                    ),
                    warmup_teacher_temp=neps.FloatParameter(
                        lower=0.001, upper=0.1, log=True, default=0.04, default_confidence="high"
                    ),
                    warmup_teacher_temp_epochs=neps.IntegerParameter(
                        lower=0, upper=50, log=False, default=0, default_confidence="high"
                    ),
                    weight_decay=neps.FloatParameter( # todo: decrease or try-except with loss=0
                        lower=0.001, upper=0.5, log=True, default=0.04, default_confidence="high"
                    ),
                    weight_decay_end=neps.FloatParameter(
                        lower=0.001, upper=0.5, log=True, default=0.4, default_confidence="high"
                    ),
                    freeze_last_layer=neps.IntegerParameter(
                        lower=0, upper=10, log=False, default=1, default_confidence="high"
                    ),
                    warmup_epochs=neps.IntegerParameter(
                        lower=0, upper=50, log=False, default=10, default_confidence="high"
                    ),
                    min_lr=neps.FloatParameter(
                        lower=1e-7, upper=1e-5, log=True, default=1e-6, default_confidence="high"
                    ),
                    drop_path_rate=neps.FloatParameter(
                        lower=0.01, upper=0.5, log=False, default=0.1, default_confidence="high"
                    ),
                    optimizer=neps.CategoricalParameter(
                        choices=['adamw', 'sgd', 'lars'], default='adamw', default_confidence="high"
                    ),
                    use_bn_in_head=neps.CategoricalParameter(
                        choices=[True, False], default=False, default_confidence="high"
                    ),
                    norm_last_layer=neps.CategoricalParameter(
                        choices=[True, False], default=True, default_confidence="high"
                    ),
                    p_horizontal_crop_1=neps.FloatParameter(
                        lower=0., upper=1., log=False, default=0.5, default_confidence="high"
                    ),
                    p_colorjitter_crop_1=neps.FloatParameter(
                        lower=0., upper=1., log=False, default=0.8, default_confidence="high"
                    ),
                    p_grayscale_crop_1=neps.FloatParameter(
                        lower=0., upper=1., log=False, default=0.2, default_confidence="high"
                    ),
                    p_horizontal_crop_2=neps.FloatParameter(
                        lower=0., upper=1., log=False, default=0.5, default_confidence="high"
                    ),
                    p_colorjitter_crop_2=neps.FloatParameter(
                        lower=0., upper=1., log=False, default=0.8, default_confidence="high"
                    ),
                    p_grayscale_crop_2=neps.FloatParameter(
                        lower=0., upper=1., log=False, default=0.2, default_confidence="high"
                    ),
                    p_horizontal_crop_3=neps.FloatParameter(
                        lower=0., upper=1., log=False, default=0.5, default_confidence="high"
                    ),
                    p_colorjitter_crop_3=neps.FloatParameter(
                        lower=0., upper=1., log=False, default=0.8, default_confidence="high"
                    ),
                    p_grayscale_crop_3=neps.FloatParameter(
                        lower=0., upper=1., log=False, default=0.2, default_confidence="high"
                    ),
                    epoch_fidelity=neps.IntegerParameter(lower=1, upper=args.epochs, is_fidelity=True),
                )
       
        #dino_neps_main = partial(dino_neps_main, args=args)
        def main():
            with Path(args.config_file_path).open('rb') as f:
                dct_to_load = pickle.load(f)
            hypers = dct_to_load['hypers']
            working_directory = dct_to_load['working_directory']
            previous_working_directory = dct_to_load['working_directory']
                
            return dino_neps_main(working_directory=working_directory, previous_working_directory=previous_working_directory,
                                  args=args, **hypers)


        def main_master(working_directory, previous_working_directory, **hypers):
            dct_to_dump = {"working_directory": working_directory, "previous_working_directory": previous_working_directory, "hypers": hypers}
            with Path(args.config_file_path).open('wb') as f:
                pickle.dump(dct_to_dump, f)
            torch.distributed.barrier()
            return main()
            
        
        def main_worker():
            torch.distributed.barrier()
            main()
            
        
        if torch.distributed.get_rank() == 0:
            neps.run(
                run_pipeline=main_master,
                pipeline_space=pipeline_space,
                working_directory=args.output_dir,
                max_evaluations_total=10000,
                max_evaluations_per_run=1,
                eta=4,
                early_stopping_rate=1,
            )
        else:
            main_worker()

    # Default DINO run
    else:
        dino_neps_main(args.output_dir, previous_working_directory=None, args=args)
