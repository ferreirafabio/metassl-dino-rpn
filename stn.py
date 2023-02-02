import math

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms.functional import resize, center_crop
from torch.autograd import Variable

N_PARAMS = {
    'affine': 6,
    'rotation': 1,
    'rotation_scale': 3,
    'rotation_scale_symmetric': 2,
    'rotation_translation': 3,
    'rotation_translation_scale': 5,
    'rotation_translation_scale_symmetric': 4,
    'scale': 2,
    'scale_symmetric': 1,
    'translation': 2,
    'translation_scale': 4,
    'translation_scale_symmetric': 3,
}

IDENT_TENSORS = {
    'affine': [1, 0, 0, 0, 1, 0],
    'rotation': [0],
    'rotation_scale': [0, 1, 1],
    'rotation_scale_symmetric': [0, 1],
    'rotation_translation': [0, 0, 0],
    'rotation_translation_scale': [0, 0, 0, 1, 1],
    'rotation_translation_scale_symmetric': [0, 0, 0, 1],
    'scale': [1, 1],
    'scale_symmetric': [1],
    'translation': [0, 0],
    'translation_scale': [0, 0, 1, 1],
    'translation_scale_symmetric': [0, 0, 1],
}


class GradientReverse(torch.autograd.Function):
    scale = 1.0

    @staticmethod
    def forward(ctx, x):
        #  autograd checks for changes in tensor to determine if backward should be called
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReverse.scale * grad_output.neg()


def grad_reverse(x, scale=1.0):
    GradientReverse.scale = scale
    return GradientReverse.apply(x)


class LocBackbone(nn.Module):
    def __init__(self, conv1_depth=32, conv2_depth=32):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(3, conv1_depth, kernel_size=3, padding=2)
        # self.conv2d_bn1 = nn.BatchNorm2d(conv1_depth)
        self.maxpool2d = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = nn.Conv2d(conv1_depth, conv2_depth, kernel_size=3, padding=2)
        # self.conv2d_bn2 = nn.BatchNorm2d(conv2_depth)
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))

    def forward(self, x):
        # xs = self.maxpool2d(F.leaky_relu(self.conv2d_bn1(self.conv2d_1(x))))
        xs = self.maxpool2d(F.leaky_relu(self.conv2d_1(x)))
        # xs = self.avgpool(F.leaky_relu(self.conv2d_bn2(self.conv2d_2(xs))))
        xs = self.avgpool(F.leaky_relu(self.conv2d_2(xs)))
        return xs


class LocHead(nn.Module):
    def __init__(self, mode, feature_dim: int):
        super().__init__()
        self.mode = mode
        self.stn_n_params = N_PARAMS[mode]
        self.feature_dim = feature_dim
        self.linear0 = nn.Linear(feature_dim, 128)
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, self.stn_n_params)

        # Initialize the weights/bias with identity transformation
        self.linear2.weight.data.zero_()
        self.linear2.bias.data.copy_(torch.tensor(IDENT_TENSORS[mode], dtype=torch.float))

    def forward(self, x):
        xs = torch.flatten(x, 1)
        xs = F.leaky_relu(self.linear0(xs))
        xs = F.leaky_relu(self.linear1(xs))
        xs = self.linear2(xs)
        return xs


class LocNet(nn.Module):
    """
    Localization Network for the Spatial Transformer Network. Consists of a ResNet-Backbone and FC-Head
    """

    def __init__(self, mode: str = 'affine', invert_gradient: bool = False,
                 num_heads: int = 1, separate_backbones: bool = False,
                 conv1_depth: int = 16, conv2_depth: int = 32, avgpool: int = 8):
        super().__init__()
        self.mode = mode
        self.invert_gradient = invert_gradient
        self.separate_backbones = separate_backbones
        self.num_heads = num_heads
        self.feature_dim = conv2_depth * avgpool ** 2

        num_backbones = num_heads if self.separate_backbones else 1

        self.backbones = nn.ModuleList(
            [LocBackbone(conv1_depth, conv2_depth) for _ in range(num_backbones)]
        )
        self.heads = nn.ModuleList(
            [LocHead(self.mode, self.feature_dim) for _ in range(self.num_heads)]
        )

    def forward(self, x):
        if self.separate_backbones:
            outputs = [h(b(x)) for b, h in zip(self.backbones, self.heads)]
        else:
            xs = self.backbones[0](x)
            outputs = [head(xs) for head in self.heads]
        if self.invert_gradient:
            outputs = [grad_reverse(theta) for theta in outputs]
        return outputs


class STN(nn.Module):
    """
    Spatial Transformer Network with a ResNet localization backbone
    """
    def __init__(self, mode: str = 'affine', invert_gradients: bool = False,
                 local_crops_number: int = 2,
                 separate_localization_net: bool = False,
                 conv1_depth: int = 32, conv2_depth: int = 32,
                 unbounded_stn: bool = False,
                 theta_norm: bool = False,
                 resolution: int = 32,
                 global_crops_scale: tuple = (0.4, 1), local_crops_scale: tuple = (0.05, 0.4),):
        super().__init__()
        self.mode = mode
        self.stn_n_params = N_PARAMS[mode]
        self.separate_localization_net = separate_localization_net
        self.invert_gradients = invert_gradients
        self.conv1_depth = conv1_depth
        self.conv2_depth = conv2_depth
        self.theta_norm = theta_norm
        self.avgpool = 8
        self.unbounded_stn = unbounded_stn
        self.local_crops_number = local_crops_number
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale

        self.res = resolution

        self.total_crops_number = 1
        # Spatial transformer localization-network
        self.localization_net = LocNet(self.mode, self.invert_gradients, self.total_crops_number,
                                       self.separate_localization_net, self.conv1_depth, self.conv2_depth, self.avgpool)

    def _get_stn_mode_theta(self, theta, x):  # Fastest
        if self.mode == 'affine':
            theta = theta if self.unbounded_stn else torch.tanh(theta)
            return theta.view(-1, 2, 3)

        out = torch.zeros([x.size(0), 2, 3], dtype=torch.float32, device=x.get_device(), requires_grad=True) + 0
        a, b, tx = [1., 0., 0.]
        c, d, ty = [0., 1., 0.]

        if 'rotation' in self.mode:
            angle = theta[:, 0]
            a = torch.cos(angle)
            b = -torch.sin(angle)
            c = torch.sin(angle)
            d = torch.cos(angle)
        if 'translation' in self.mode:
            x, y = (0, 1) if self.mode.startswith('translation') else (1, 2)
            tx = theta[:, x] if self.unbounded_stn else torch.tanh(theta[:, x])
            ty = theta[:, y] if self.unbounded_stn else torch.tanh(theta[:, y])
        if 'scale' in self.mode:
            if 'symmetric' in self.mode:
                sx = theta[:, -1] if self.unbounded_stn else torch.tanh(theta[:, -1])
                sy = theta[:, -1] if self.unbounded_stn else torch.tanh(theta[:, -1])
            else:
                sx = theta[:, -2] if self.unbounded_stn else torch.tanh(theta[:, -2])
                sy = theta[:, -1] if self.unbounded_stn else torch.tanh(theta[:, -1])
            a *= sx
            b *= sx
            c *= sy
            d *= sy

        out[:, 0, 0] = a
        out[:, 0, 1] = b
        out[:, 0, 2] = tx
        out[:, 1, 0] = c
        out[:, 1, 1] = d
        out[:, 1, 2] = ty
        return out

    def forward(self, x):
        theta_params = self.localization_net(x)[0]

        theta = self._get_stn_mode_theta(theta_params, x)

        if self.theta_norm:
            theta = theta / torch.linalg.norm(theta, ord=1, dim=2, keepdim=True).clamp(min=1)

        align_corners = True
        grid = F.affine_grid(theta, size=list(x.size()[:2]) + [self.res, self.res], align_corners=align_corners)
        crop = F.grid_sample(x, grid, align_corners=align_corners)

        return crop, theta


class AugmentationNetwork(nn.Module):
    def __init__(self, transform_net: STN, resize_input: bool = False, resize_size: int = 512):
        super().__init__()
        print("Initializing Augmentation Network")
        self.transform_net = transform_net
        self.resize_input = resize_input
        self.resize_size = resize_size

    def forward(self, x):
        # if we get a tensor as input, simply pass it to the STN
        if isinstance(x, torch.Tensor):
            return self.transform_net(x)

        # otherwise the input should be a list of PIL images, e.g. uncropped ImageNet dataset
        if not isinstance(x, list):
            x = [x]
        for idx, img in enumerate(x):
            if self.resize_input and max(img.size()) > self.resize_size:
                img = resize(img, size=[self.resize_size, ], max_size=self.resize_size+1)
            img = img.unsqueeze(0)
            x[idx] = img

        num_crops = self.transform_net.local_crops_number + 2
        views = [[] for _ in range(num_crops)]
        thetas = [[] for _ in range(num_crops)]
        for img in x:
            views_net, thetas_net = self.transform_net(img)

            for idx, (view, theta) in enumerate(zip(views_net, thetas_net)):
                views[idx].append(view)
                thetas[idx].append(theta)

        views = [torch.cat(view) for view in views]
        thetas = [torch.cat(theta) for theta in thetas]

        return views, thetas

    def mc_forward(self, x):
        if self.resize_input:
            # If we have list of images with varying resolution, we need to transform them individually
            if not isinstance(x, list):
                x = [x]
            # Prepare input for STN, additionally resize input to avoid OOM error
            for idx, img in enumerate(x):
                if max(img.size()) > self.resize_size:
                    img = resize(img, size=[self.resize_size, ], max_size=self.resize_size+1)
                if img.size(-1) != img.size(-2):
                    img = center_crop(img, min(img.size()[-2:]))
                img = img.unsqueeze(0)
                x[idx] = img
            # Concat inputs with same size to improve inference/training time
            idx_crops = torch.cumsum(torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1], 0)
            start_idx = 0
            # Init for output lists
            num_crops = self.transform_net.local_crops_number + 2
            views = [[] for _ in range(num_crops)]
            thetas = [[] for _ in range(num_crops)]
            idx_merged = [torch.zeros(1, dtype=idx_crops.dtype).cuda() for _ in range(dist.get_world_size())]
            torch.nn.functional.pad(idx_crops, (0, len(x) - len(idx_crops)), value=0)
            dist.all_gather(idx_merged, idx_crops)
            idx_crops = torch.cat(idx_merged).unique()
            for end_idx in idx_crops:
                views_net, thetas_net = self.transform_net(torch.cat(x[start_idx: end_idx]))

                for idx, (view, theta) in enumerate(zip(views_net, thetas_net)):
                    views[idx].append(view)
                    thetas[idx].append(theta)

                start_idx = end_idx

            views = [torch.cat(view) for view in views]
            thetas = [torch.cat(theta) for theta in thetas]

            return views, thetas

        else:
            if isinstance(x, list):
                x = torch.stack(x)
            return self.transform_net(x)
