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
import copy
import datetime
import json
import math
import os
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from penalties import HISTLoss, SIMLoss, ThetaLoss
from stn import AugmentationNetwork, STN
from utils import custom_collate, SummaryWriterCustom
from vision_transformer import DINOHead

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))

penalty_dict = {
    "simloss": SIMLoss,
    "histloss": HISTLoss,
    "thetaloss": ThetaLoss,
}


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
        mixed precision training (--use_fp16 false) to avoid instabilities.""")
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
                        choices=['adamw', 'sgd', 'lars'],
                        help="""Type of optimizer. We recommend using adamw with ViTs.""")
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

    # Misc
    parser.add_argument("--dataset", default="ImageNet", type=str, choices=["ImageNet", "CIFAR10"],
                        help="Specify the name of your dataset. Choose from: ImageNet, CIFAR10")
    parser.add_argument('--data_path', default='/path/to/dataset/train/', type=str,
                        help='Please specify path to the training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=3, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--world_size", default=8, type=int, help="default is for NEPS mode with DDP, so 8.")
    parser.add_argument("--gpu", default=8, type=int, help="default is for NEPS mode with DDP, so 8 GPUs.")
    parser.add_argument('--config_file_path', help="Should be set to a path that does not exist.")

    # STN
    parser.add_argument("--invert_stn_gradients", default=False, type=utils.bool_flag,
                        help="Set this flag to invert the gradients used to learn the STN")
    parser.add_argument("--use_stn_optimizer", default=False, type=utils.bool_flag,
                        help="Set this flag to use a separate optimizer for the STN parameters; "
                             "annealed with cosine and no warmup")
    parser.add_argument('--stn_mode', default='affine', type=str,
                        help='Determines the STN mode (choose from: affine, translation, scale, rotation, '
                             'rotation_scale, translation_scale, rotation_translation, rotation_translation_scale')
    parser.add_argument("--stn_lr", default=5e-4, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training) of the STN optimizer. The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--separate_localization_net", default=False, type=utils.bool_flag,
                        help="Set this flag to use a separate localization network for each head.")
    parser.add_argument("--summary_writer_freq", default=5000, type=int,
                        help="Defines the number of iterations the summary writer will write output.")
    parser.add_argument("--grad_check_freq", default=5000, type=int,
                        help="Defines the number of iterations the current tensor grad of the global 1 localization head is printed to stdout.")
    parser.add_argument("--summary_plot_size", default=16, type=int,
                        help="Defines the number of samples to show in the summary writer.")
    parser.add_argument('--stn_pretrained_weights', default='', type=str,
                        help="Path to pretrained weights of the STN network. If specified, the STN is not trained and used to pre-process images solely.")
    parser.add_argument("--deep_loc_net", default=False, type=utils.bool_flag,
                        help="Set this flag to use a deep loc net (default: False).")
    parser.add_argument("--use_one_res", default=False, type=utils.bool_flag,
                        help="Set this flag to only use one target resolution (128x128) after STN transformation (instead of 224x and 96x)")
    parser.add_argument("--use_unbounded_stn", default=False, type=utils.bool_flag,
                        help="Set this flag to not use a tanh in the last STN layer (default: use bounded STN).")
    parser.add_argument("--stn_warmup_epochs", default=0, type=int,
                        help="Specifies the number of warmup epochs for the STN (default: 0).")
    parser.add_argument("--stn_conv1_depth", default=32, type=int,
                        help="Specifies the number of feature maps of conv1 for the STN localization network (default: 32).")
    parser.add_argument("--stn_conv2_depth", default=32, type=int,
                        help="Specifies the number of feature maps of conv2 for the STN localization network (default: 32).")
    parser.add_argument("--resize_input", default=False, type=utils.bool_flag,
                        help="Set this flag to resize the images of the dataset, can be useful for datasets with varying resolutions")
    parser.add_argument("--stn_theta_norm", default=False, type=utils.bool_flag,
                        help="Set this flag to normalize 'theta' in the STN before passing to affine_grid(theta, ...). Fixes the problem with cropping of the images (black regions)")
    parser.add_argument("--use_similarity_penalty", default=False, type=utils.bool_flag,
                        help="Set this flag to add a penalty term to the loss. Similarity between input and output image of STN.")
    parser.add_argument("--similarity_measure", default="simloss", type=str, choices=list(penalty_dict.keys()),
                        help="Specify the name if the similarity to use.")
    parser.add_argument("--invert_penalty", default=False, type=utils.bool_flag,
                        help="Invert the Similarity Penalty Loss.")
    parser.add_argument("--min_sim_pen", default=1.0, type=float,
                        help="Set the minimal similarity value of SIMLoss. Sets the loss to 0 if the similarity is higher.")
    parser.add_argument("--global_res", default=224, type=int, help="Global output resolution of the STN")
    parser.add_argument("--local_res", default=96, type=int, help="Local output resolution of the STN")
    parser.add_argument("--one_res", default=128, type=int,
                        help="Single output resolution of the STN. Use with '--use_one_res'.")

    # tests
    parser.add_argument("--test_mode", default=False, type=utils.bool_flag, help="Set this flag to activate test mode.")

    return parser


def train_dino(args):
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    with (Path(args.output_dir) / "settings.json").open("w") as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    # ============ preparing data ... ============
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate,
    )

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
        print(f"Unknown architecture: {args.arch}")

    transform_net = STN(stn_mode=args.stn_mode,
                        separate_localization_net=args.separate_localization_net,
                        invert_stn_gradients=args.invert_stn_gradients,
                        deep_loc_net=args.deep_loc_net,
                        use_one_res=args.use_one_res,
                        use_unbounded_stn=args.use_unbounded_stn,
                        conv1_depth=args.stn_conv1_depth,
                        conv2_depth=args.stn_conv2_depth,
                        resize_input=args.resize_input,
                        theta_norm=args.stn_theta_norm,
                        global_res=args.global_res,
                        local_res=args.local_res,
                        one_res=args.one_res
                        )
    stn = AugmentationNetwork(transform_net=transform_net)

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
    student, teacher, stn = student.cuda(), teacher.cuda(), stn.cuda()

    # teacher and student start with the same weights
    teacher.load_state_dict(student.state_dict())
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

    if args.dataset == "CIFAR10":
        min_res = min(32, args.local_res)
    else:
        min_res = args.local_res
    if args.use_one_res:
        min_res = min(min_res, args.one_res)
    sim_loss = None
    if args.use_similarity_penalty:
        # argz = {
        #     'invert': args.invert_penalty,
        #     'min_sim': args.min_sim_pen,
        #     'resolution': 32,
        #     'exponent': 2,
        #     'bins': 100,
        # }
        Loss = penalty_dict[args.similarity_measure]
        sim_loss = Loss(
            invert=args.invert_penalty,
            min_sim=args.min_sim_pen,
            resolution=32,
            exponent=2,
            bins=100,
            device=torch.device('cuda')
        ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)

    # ============ handling STN optimization ============
    stn_optimizer = None
    use_pretrained_stn = os.path.isfile(args.stn_pretrained_weights)
    if use_pretrained_stn:
        utils.load_stn_pretrained_weights(stn, args.stn_pretrained_weights)

        for p in stn.parameters():
            p.requires_grad = False

    if not args.use_stn_optimizer and not use_pretrained_stn:
        # add to regularized param group
        # student_params = params_groups[0]['params']
        # all_params = student_params + list(stn.parameters())
        # params_groups[0]['params'] = all_params

        # do not use wd scheduling of STN params
        student_params = params_groups[1]['params']
        all_params = student_params + list(stn.parameters())
        params_groups[1]['params'] = all_params

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        if args.use_stn_optimizer and not use_pretrained_stn:
            stn_optimizer = torch.optim.AdamW(list(stn.parameters()))  # lr is set by scheduler
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
        if args.use_stn_optimizer and not use_pretrained_stn:
            stn_optimizer = torch.optim.SGD(list(stn.parameters()), lr=0)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
        if args.use_stn_optimizer and not use_pretrained_stn:
            stn_optimizer = torch.optim.LARS(list(stn.parameters()))  # lr is set by scheduler

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

    stn_lr_schedule = None
    if stn_optimizer and not use_pretrained_stn:
        stn_lr_schedule = utils.cosine_scheduler(
            args.stn_lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
            args.min_lr,
            args.epochs, len(data_loader),
            warmup_epochs=args.stn_warmup_epochs,
        )

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
        stn=stn,
        stn_optimizer=stn_optimizer,
    )

    start_epoch = to_restore["epoch"]

    if use_pretrained_stn:
        for p in stn.parameters():
            p.requires_grad = False

    summary_writer = SummaryWriterCustom(Path(args.output_dir) / "summary", plot_size=args.summary_plot_size)

    start_time = time.time()
    print("Starting DINO training !")

    end_epoch = args.epochs

    for epoch in range(start_epoch, end_epoch):
        # data_loader.sampler.set_epoch(epoch)
        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, dino_loss, sim_loss,
                                      data_loader, optimizer, stn_optimizer, lr_schedule, wd_schedule, stn_lr_schedule,
                                      momentum_schedule,
                                      epoch, fp16_scaler, stn, use_pretrained_stn, args, summary_writer)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
            'stn': stn.state_dict(),
            'stn_optimizer': stn_optimizer.state_dict() if stn_optimizer else None,
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


def train_one_epoch(student, teacher, dino_loss, sim_loss, data_loader,
                    optimizer, stn_optimizer, lr_schedule, wd_schedule, stn_lr_schedule, momentum_schedule, epoch,
                    fp16_scaler, stn, use_pretrained_stn, args, summary_writer):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        if args.use_stn_optimizer and not use_pretrained_stn:
            for i, param_group in enumerate(stn_optimizer.param_groups):
                param_group["lr"] = stn_lr_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]

        if it % args.summary_writer_freq == 0:
            uncropped_images = copy.deepcopy(images)

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            stn_images, theta = stn(images)
            penalty = 0
            if args.use_similarity_penalty:
                penalty = sim_loss(images=stn_images, theta=theta, target=images)

            if it % args.summary_writer_freq == 0:
                summary_writer.write_image_grid(tag="images", images=stn_images, original_images=uncropped_images,
                                                epoch=epoch, global_step=it)
                summary_writer.write_theta_heatmap(tag="theta_g1", theta=stn.transform_net.affine_matrix_g1,
                                                   epoch=epoch, global_step=it)
                summary_writer.write_theta_heatmap(tag="theta_g2", theta=stn.transform_net.affine_matrix_g2,
                                                   epoch=epoch, global_step=it)
                summary_writer.write_theta_heatmap(tag="theta_l1", theta=stn.transform_net.affine_matrix_l1,
                                                   epoch=epoch, global_step=it)
                summary_writer.write_theta_heatmap(tag="theta_l2", theta=stn.transform_net.affine_matrix_l2,
                                                   epoch=epoch, global_step=it)

                theta_g_euc_norm = np.linalg.norm(np.double(
                    stn.transform_net.affine_matrix_g2[0] - stn.transform_net.affine_matrix_g1[0]), 2)
                theta_l_euc_norm = np.linalg.norm(np.double(
                    stn.transform_net.affine_matrix_l2[0] - stn.transform_net.affine_matrix_l1[0]), 2)
                summary_writer.write_scalar(tag="theta local eucl. norm.", scalar_value=theta_l_euc_norm,
                                            global_step=it)
                summary_writer.write_scalar(tag="theta global eucl. norm.", scalar_value=theta_g_euc_norm,
                                            global_step=it)

                uncropped_images = None

            # continue
            teacher_output = teacher(stn_images[:2])  # only the 2 global views pass through the teacher
            student_output = student(stn_images)
            loss = dino_loss(student_output, teacher_output, epoch) + penalty

            summary_writer.write_scalar(tag="loss", scalar_value=loss.item(), global_step=it)
            summary_writer.write_scalar(tag="lr", scalar_value=optimizer.param_groups[0]["lr"], global_step=it)
            if args.use_stn_optimizer and not use_pretrained_stn:
                summary_writer.write_scalar(tag="lr stn", scalar_value=stn_optimizer.param_groups[0]["lr"],
                                            global_step=it)
            summary_writer.write_scalar(tag="weight decay", scalar_value=optimizer.param_groups[0]["weight_decay"],
                                        global_step=it)

            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(penalty)
            raise ValueError("Loss value is invalid.")

        if args.use_stn_optimizer and not use_pretrained_stn:
            stn_optimizer.zero_grad()

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:

            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)

            optimizer.step()

            if args.use_stn_optimizer and not use_pretrained_stn:
                stn_optimizer.step()

        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)

            if args.use_stn_optimizer and not use_pretrained_stn:
                fp16_scaler.step(stn_optimizer)
            fp16_scaler.update()

        if it % args.grad_check_freq == 0:
            print(stn.transform_net.fc_localization_local1.linear2.weight)
            if args.separate_localization_net:
                print(stn.transform_net.localization_net_g1.conv2d_2.weight)
            else:
                print(stn.transform_net.localization_net.conv2d_2.weight)
            print("-------------------------sanity check local grads-------------------------------")
            print(stn.transform_net.fc_localization_local1.linear2.weight.grad)
            print("-------------------------sanity check global grads-------------------------------")
            print(stn.transform_net.fc_localization_global1.linear2.weight.grad)
            if args.separate_localization_net:
                print(stn.transform_net.localization_net_g1.conv2d_2.weight.grad)
            else:
                print(stn.transform_net.localization_net.conv2d_2.weight.grad)
            print(f"CUDA MAX MEM:           {torch.cuda.max_memory_allocated()}")
            print(f"CUDA MEM ALLOCATED:     {torch.cuda.memory_allocated()}")

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if args.use_stn_optimizer and not use_pretrained_stn:
            metric_logger.update(lrstn=stn_optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
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
        batch_center = batch_center / len(teacher_output)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
