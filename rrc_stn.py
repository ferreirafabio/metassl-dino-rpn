import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms.functional import crop, resize
from torch.autograd import Variable

N_PARAMS = {
    'affine': 6,
    'rotation': 1,
    'rotation_translation': 3,
    'scale': 1,
    'scaled_rotation': 2,
    'scaled_translation': 3,
    'scaled_rotation_translation': 4,
    'shear': 2,
    'translation': 2,
}
# needed for the spatial transformer net

IDENT_TENSORS = {
    'affine': [1, 0, 0, 0, 1, 0],
    'rotation': [0],
    'rotation_translation': [0, 0, 0],
    'scale': [1],
    'scaled_rotation': [1, 0],
    'scaled_rotation_translation': [1, 0, 0, 0],
    'scaled_translation': [1, 0, 0],
    'shear': [1, 1],
    'translation': [0, 0],
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


class Clamp2(torch.autograd.Function):
    """
    Clamps the given tensor in the given range on both sides of zero (negative and positive).
    Given values:
        min_val = 0.5
        max_val = 1
        tensor = [-2, -1, -0.75, -0.25, 0?, 0.25, 0.75, 1, 2]
    Result -> [-1, -1, -0.75, -0.5, -0.5?0?0.5, 0.5, 0.75, 1, 1]
    """
    @staticmethod
    def forward(ctx, x, min_val, max_val):
        # ctx.save_for_backward(x)
        # TODO: at the moment 0 is always assigned to the positive range, before it was 0 because of sign
        # But the clamping assigned the min_val, but sign(0) = 0, min_val * 0 = 0
        # Another solution could be to add a small value, like 0.00001 to theta
        sign = x.sign()
        sign[sign == 0] = 1
        ctx._mask = (x.abs().ge(min_val) * x.abs().le(max_val))
        # return x.abs().clamp(min_val, max_val) * x.sign()
        return x.abs().clamp(min_val, max_val) * sign

    def backward(ctx, grad_output):
        mask = Variable(ctx._mask.type_as(grad_output.data))
        # x, = ctx.saved_variables
        # Not sure whether x.sign() is needed here
        # I dont think so, because we keep the sign before clamping in the forward pass
        # return grad_output * mask * x.sign(), None, None
        return grad_output * mask, None, None


def clamp2(x, min_val, max_val):
    return Clamp2.apply(x, min_val, max_val)


class LocBackbone(nn.Module):
    def __init__(self, conv1_depth=32, conv2_depth=32):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(3, conv1_depth, kernel_size=3, padding=2)
        self.conv2d_bn1 = nn.BatchNorm2d(conv1_depth)
        self.maxpool2d = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = nn.Conv2d(conv1_depth, conv2_depth, kernel_size=3, padding=2)
        self.conv2d_bn2 = nn.BatchNorm2d(conv2_depth)
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))

    def forward(self, x):
        xs = self.maxpool2d(F.leaky_relu(self.conv2d_bn1(self.conv2d_1(x))))
        xs = self.avgpool(F.leaky_relu(self.conv2d_bn2(self.conv2d_2(xs))))
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
                 num_heads: int = 4, separate_backbones: bool = False,
                 conv1_depth: int = 16, conv2_depth: int = 32, avgpool: int = 8):
        super().__init__()
        self.mode = mode
        self.invert_gradient = invert_gradient
        self.separate_backbones = separate_backbones
        self.num_heads = num_heads
        self.feature_dim = conv2_depth * avgpool ** 2

        num_backbones = num_heads if self.separate_backbones else 1
        backbones = []
        for _ in range(num_backbones):
            module = LocBackbone(conv1_depth, conv2_depth)
            backbones.append(module)
        self.backbones = nn.ModuleList(backbones)

        heads = []
        for _ in range(self.num_heads):
            module = LocHead(self.mode, self.feature_dim)
            heads.append(module)
        self.heads = nn.ModuleList(heads)

    def forward(self, x):
        thetas = []
        if self.separate_backbones:
            for backbone, head in zip(self.backbones, self.heads):
                theta = head(backbone(x))
                thetas.append(theta)
        else:
            xs = self.backbones[0](x)
            for head in self.heads:
                theta = head(xs)
                thetas.append(theta)
        if self.invert_gradient:
            thetas = [grad_reverse(theta) for theta in thetas]
        return thetas


class STN(nn.Module):
    """
    Spatial Transformer Network with a ResNet localization backbone
    """
    def __init__(self, mode: str = 'affine', invert_gradients: bool = False,
                 local_crops_number: int = 2,
                 separate_localization_net: bool = False,
                 conv1_depth: int = 32, conv2_depth: int = 32,
                 resize_input: bool = False,
                 theta_norm: bool = False,
                 resolution: tuple = (224, 96),
                 global_crops_scale: tuple = (0.4, 1), local_crops_scale: tuple = (0.05, 0.4)
                 ):
        super().__init__()
        self.mode = mode
        self.stn_n_params = N_PARAMS[mode]
        self.local_crops_number = local_crops_number
        self.separate_localization_net = separate_localization_net
        self.invert_gradients = invert_gradients
        self.conv1_depth = conv1_depth
        self.conv2_depth = conv2_depth
        self.resize_input = resize_input
        self.theta_norm = theta_norm
        self.avgpool = 8
        self.use_unbounded_stn = False
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale

        assert len(resolution) < 3, f"resolution parameter should be of length 1 or 2, but {len(resolution)} with {resolution} is given."
        if len(resolution) == 1:
            self.global_res = self.local_res = resolution[0]
        else:
            self.global_res, self.local_res = resolution

        self.affine_matrix_g1 = None
        self.affine_matrix_g2 = None
        self.affine_matrix_l1 = None
        self.affine_matrix_l2 = None

        self.total_crops_number = 2 + self.local_crops_number
        # Spatial transformer localization-network
        self.localization_net = LocNet(self.mode, self.invert_gradients, self.total_crops_number,
                                       self.separate_localization_net, self.conv1_depth, self.conv2_depth, self.avgpool)

        self.gmin_scale = math.pow(self.global_crops_scale[0], .25)
        self.gmax_scale = math.pow(self.global_crops_scale[1], .25)
        self.gmin_txy = self.gmin_scale - 1
        self.gmax_txy = 1 - self.gmin_scale

        self.lmin_scale = math.pow(self.local_crops_scale[0], .25)
        self.lmax_scale = math.pow(self.local_crops_scale[1], .25)
        self.lmin_txy = 1 - self.lmax_scale
        self.lmax_txy = 1 - self.lmin_scale

    def _get_stn_mode_theta(self, theta, x, crop_mode: str = 'global'):
        if self.mode == 'affine':
            theta = theta.clone()
            theta[:, 1:] = theta[:, 1:] if self.use_unbounded_stn else torch.clone(
                torch.tanh(theta[:, 1:]))  # optionally bound everything except for the angle at [:, 0]
            theta_new = theta.view(-1, 2, 3)
        else:
            theta_new = torch.zeros([x.size(0), 2, 3], dtype=torch.float32, device=x.get_device(), requires_grad=True)
            theta_new = theta_new + 0
            theta_new[:, 0, 0] = 1.0
            theta_new[:, 1, 1] = 1.0
            if self.mode == "rotation":
                angle = theta[:, 0]
                theta_new[:, 0, 0] = torch.cos(angle)
                theta_new[:, 0, 1] = -torch.sin(angle)
                theta_new[:, 1, 0] = torch.sin(angle)
                theta_new[:, 1, 1] = torch.cos(angle)
            elif self.mode == "scaled_translation":
                if crop_mode == 'global':
                    scale = theta[:, 0].clamp(self.gmin_scale, self.gmax_scale).view(-1, 1, 1)
                    # scale = clamp2(theta[:, 0], self.gmin_scale, self.gmax_scale).view(-1, 1, 1)
                    txy = theta[:, 1:].clamp(-self.gmax_txy, self.gmax_txy)
                    # tx = theta[:, 1].clamp(-self.gmax_txy, self.gmax_txy)
                    # ty = theta[:, 2].clamp(-self.gmax_txy, self.gmax_txy)
                else:
                    scale = theta[:, 0].clamp(self.lmin_scale, self.lmax_scale).view(-1, 1, 1)  # simpler version that does not allow horizontal and vertical flipping
                    txy = theta[:, 1:].clamp(-self.lmax_txy, self.lmax_txy)
                    # scale = clamp2(theta[:, 0], self.lmin_scale, self.lmax_scale).view(-1, 1, 1)
                    # txy = clamp2(theta[:, 1:], self.lmin_txy, self.lmax_txy)  # oneliner
                    # tx = clamp2(theta[:, 1], self.lmin_txy, self.lmax_txy)
                    # ty = clamp2(theta[:, 2], self.lmin_txy, self.lmax_txy)
                theta_new = theta_new * scale
                theta_new[:, :, 2] = txy
                # theta_new[:, 0, 2] = tx
                # theta_new[:, 1, 2] = ty
        return theta_new

    def forward(self, x):
        theta_params = self.localization_net(x)

        # 2 because of 2 global crops/views
        thetas = [self._get_stn_mode_theta(params, x, 'global') for params in theta_params[:2]] + \
                 [self._get_stn_mode_theta(params, x, 'local') for params in theta_params[2:]]

        if self.theta_norm:
            thetas = [theta / torch.linalg.norm(theta, ord=1, dim=2, keepdim=True).clamp(min=1) for theta in thetas]

        self.affine_matrix_g1 = thetas[0].cpu().detach().numpy()
        self.affine_matrix_g2 = thetas[1].cpu().detach().numpy()
        self.affine_matrix_l1 = thetas[2].cpu().detach().numpy()
        self.affine_matrix_l2 = thetas[3].cpu().detach().numpy()

        align_corners = True
        crops = []
        for theta in thetas[:2]:
            grid = F.affine_grid(theta, size=list(x.size()[:2]) + [self.global_res, self.global_res], align_corners=align_corners)
            crop = F.grid_sample(x, grid, align_corners=align_corners)
            crops.append(crop)
        for theta in thetas[2:]:
            grid = F.affine_grid(theta, size=list(x.size()[:2]) + [self.local_res, self.local_res], align_corners=align_corners)
            crop = F.grid_sample(x, grid, align_corners=align_corners)
            crops.append(crop)

        return crops, thetas


class AugmentationNetwork(nn.Module):
    def __init__(self, transform_net: STN):
        super().__init__()
        print("Initializing Augmentation Network")
        self.transform_net = transform_net

    def forward(self, imgs):
        if self.transform_net.resize_input:
            # If we have list of images with varying resolution, we need to transform them individually
            global_views1_list, global_views2_list, local_views1_list, local_views2_list = [], [], [], []
            for img in imgs:
                img = torch.unsqueeze(img, 0)
                try:
                    if img.size(2) > 700 or img.size(3) > 700:
                        img = resize(img, size=700, max_size=701)
                except Exception as e:
                    print(e)

                global_local_views, _ = self.transform_net(img)

                g1_augmented = torch.squeeze(global_local_views[0], 0)
                g2_augmented = torch.squeeze(global_local_views[1], 0)
                l1_augmented = torch.squeeze(global_local_views[2], 0)
                l2_augmented = torch.squeeze(global_local_views[3], 0)

                global_views1_list.append(g1_augmented)
                global_views2_list.append(g2_augmented)
                local_views1_list.append(l1_augmented)
                local_views2_list.append(l2_augmented)

            global_views1 = torch.stack(global_views1_list, 0)
            global_views2 = torch.stack(global_views2_list, 0)
            local_views1 = torch.stack(local_views1_list, 0)
            local_views2 = torch.stack(local_views2_list, 0)

            del global_views1_list, global_views2_list, local_views1_list, local_views2_list

            return [global_views1, global_views2, local_views1, local_views2], []

        else:
            if isinstance(imgs, list):
                imgs = torch.stack(imgs, dim=0)
            return self.transform_net(imgs)
