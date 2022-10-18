# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from utils import resnet9
from torchvision.transforms.functional import crop, resize
from torchvision import transforms
import utils
import kornia

# from torchsummary import summary


# needed for the spatial transformer net
N_PARAMS = {
        'affine': 6,
        'translation': 2,
        'rotation': 1,
        'scale': 2,
        'shear': 2,
        'rotation_scale': 3,
        'translation_scale': 4,
        'rotation_translation': 3,
        'rotation_translation_scale': 5
    }


class GradientReverse(torch.autograd.Function):
    scale = 1.0
    
    @staticmethod
    def forward(ctx, x):
        #  autograd checks for changes in tensor to determine if backward should be called
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        print(f"normal grads: {grad_output}")
        print(f"negative grads: {grad_output.neg()}")
        print(grad_output.shape)
        return GradientReverse.scale * grad_output.neg()


def grad_reverse(x, scale=1.0):
    GradientReverse.scale = scale
    return GradientReverse.apply(x)


class ResNetRPN(nn.Module):
    def __init__(self, backbone='resnet18', out_dim=256, invert_rpn_gradients=False):
        super().__init__()
        
        self.invert_rpn_gradients = invert_rpn_gradients
        
        if backbone == 'resnet18':
            backbone = resnet18(pretrained=False)
            # summary(backbone.cuda(), (3, 224, 224))
        elif backbone == 'resnet9':
            backbone = resnet9(pretrained=False)
            print("resnet9 backbone: ", backbone)
            # summary(backbone.cuda(), (3, 224, 224))
        elif backbone == 'resnet34':
            backbone = resnet34(pretrained=False)
        elif backbone == 'resnet50':
            backbone = resnet50(pretrained=False)
        elif backbone == 'resnet101':
            backbone = resnet101(pretrained=False)
        else:  # backbone == 'resnet152':
            backbone = resnet152(pretrained=False)
        # if backbone_path:
        #     backbone.load_state_dict(torch.load(backbone_path))

        backbone.fc = nn.Linear(512, out_dim)
        torch.nn.init.xavier_uniform_(backbone.fc.weight)
        self.backbone = backbone

    def forward(self, x, invert_rpn_gradients):
        if invert_rpn_gradients:
            x = grad_reverse(x)
            
        x = self.backbone(x)
        return x

    
class LocalizationNet(nn.Module):
    def __init__(self, conv1_depth=16, conv2_depth=32, deep=False):
        super().__init__()
        
        self.deep = deep
        self.conv2d_1 = nn.Conv2d(3, conv1_depth, kernel_size=3, padding=2)
        self.maxpool2d = nn.MaxPool2d(2, stride=2)
        if self.deep:
            self.conv2d_deep = nn.Conv2d(conv1_depth, conv1_depth, kernel_size=3, padding=2)
            
        self.conv2d_2 = nn.Conv2d(conv1_depth, conv2_depth, kernel_size=3, padding=2)
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        
    def forward(self, x, invert_rpn_gradients):
        if invert_rpn_gradients:
            x = grad_reverse(x)
            
        x = self.maxpool2d(F.leaky_relu(self.conv2d_1(x)))
        if self.deep:
            x = self.maxpool2d(F.leaky_relu(self.conv2d_deep(x)))
            x = self.maxpool2d(F.leaky_relu(self.conv2d_deep(x)))
        x = self.avgpool(F.leaky_relu(self.conv2d_2(x)))
        return x


class LocHead(nn.Module):
    def __init__(self, stn_mode, conv2_depth, deep_loc_net=False):
        super().__init__()
        
        self.stn_n_params = N_PARAMS[stn_mode]
        self.deep_loc_net = deep_loc_net
        
        self.linear0 = nn.Linear(8 * 8 * conv2_depth, 256 if deep_loc_net else 128)
        self.linear1 = nn.Linear(256 if deep_loc_net else 128, 32)
        self.linear2 = nn.Linear(32, self.stn_n_params)
    
    def forward(self, x, invert_rpn_gradients):
        if invert_rpn_gradients:
            x = grad_reverse(x)
        
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.linear0(x))
        x = F.leaky_relu(self.linear1(x))
        x = self.linear2(x)
        return x
    

class STN(nn.Module):
    """"
    Spatial Transformer Network with a ResNet localization backbone
    """""
    def __init__(self, backbone="resnet18", stn_mode='affine', invert_rpn_gradients=False, separate_localization_net=False, deep_loc_net=False):
        super(STN, self).__init__()
        self.stn_mode = stn_mode
        self.stn_n_params = N_PARAMS[stn_mode]
        self.invert_rpn_gradients = invert_rpn_gradients
        self.separate_localization_net = separate_localization_net
        self.deep_loc_net = deep_loc_net
        self.affine_matrix_g1 = None
        self.affine_matrix_g2 = None
        self.affine_matrix_l1 = None
        self.affine_matrix_l2 = None
        
        # Spatial transformer localization-network
        if self.separate_localization_net:
            conv1_depth = 16
            conv2_depth = 8
            self.localization_net_g1 = LocalizationNet(conv1_depth=conv1_depth, conv2_depth=conv2_depth)
            self.localization_net_g2 = LocalizationNet(conv1_depth=conv1_depth, conv2_depth=conv2_depth)
            self.localization_net_l1 = LocalizationNet(conv1_depth=conv1_depth, conv2_depth=conv2_depth)
            self.localization_net_l2 = LocalizationNet(conv1_depth=conv1_depth, conv2_depth=conv2_depth)
        else:
            if self.deep_loc_net:
                conv1_depth = 32
                conv2_depth = 64
                self.localization_net = LocalizationNet(conv1_depth=conv1_depth, conv2_depth=conv2_depth, deep=True)
            else:
                conv1_depth = 32
                conv2_depth = 16
                self.localization_net = LocalizationNet(conv1_depth=conv1_depth, conv2_depth=conv2_depth, deep=False)

        # Regressors for the 3 * 2 affine matrix
        self.fc_localization_global1 = LocHead(stn_mode=stn_mode, conv2_depth=conv2_depth, deep_loc_net=self.deep_loc_net)
        self.fc_localization_global2 = LocHead(stn_mode=stn_mode, conv2_depth=conv2_depth, deep_loc_net=self.deep_loc_net)
        self.fc_localization_local1 = LocHead(stn_mode=stn_mode, conv2_depth=conv2_depth, deep_loc_net=self.deep_loc_net)
        self.fc_localization_local2 = LocHead(stn_mode=stn_mode, conv2_depth=conv2_depth, deep_loc_net=self.deep_loc_net)
        
        # Initialize the weights/bias with identity transformation
        self.fc_localization_global1.linear2.weight.data.zero_()
        self.fc_localization_global2.linear2.weight.data.zero_()
        self.fc_localization_local1.linear2.weight.data.zero_()
        self.fc_localization_local2.linear2.weight.data.zero_()
        
        if self.stn_mode == 'affine':
            self.fc_localization_global1.linear2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
            self.fc_localization_global2.linear2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
            self.fc_localization_local1.linear2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
            self.fc_localization_local2.linear2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
            
        elif self.stn_mode in ['translation', 'shear']:
            self.fc_localization_global1.linear2.bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
            self.fc_localization_global2.linear2.bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
            self.fc_localization_local1.linear2.bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
            self.fc_localization_local2.linear2.bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
        elif self.stn_mode == 'scale':
            self.fc_localization_global1.linear2.bias.data.copy_(torch.tensor([1, 1], dtype=torch.float))
            self.fc_localization_global2.linear2.bias.data.copy_(torch.tensor([1, 1], dtype=torch.float))
            self.fc_localization_local1.linear2.bias.data.copy_(torch.tensor([1, 1], dtype=torch.float))
            self.fc_localization_local2.linear2.bias.data.copy_(torch.tensor([1, 1], dtype=torch.float))
        elif self.stn_mode == 'rotation':
            self.fc_localization_global1.linear2.bias.data.copy_(torch.tensor([0], dtype=torch.float))
            self.fc_localization_global2.linear2.bias.data.copy_(torch.tensor([0], dtype=torch.float))
            self.fc_localization_local1.linear2.bias.data.copy_(torch.tensor([0], dtype=torch.float))
            self.fc_localization_local2.linear2.bias.data.copy_(torch.tensor([0], dtype=torch.float))
        elif self.stn_mode == 'rotation_scale':
            self.fc_localization_global1.linear2.bias.data.copy_(torch.tensor([0, 1, 1], dtype=torch.float))
            self.fc_localization_global2.linear2.bias.data.copy_(torch.tensor([0, 1, 1], dtype=torch.float))
            self.fc_localization_local1.linear2.bias.data.copy_(torch.tensor([0, 1, 1], dtype=torch.float))
            self.fc_localization_local2.linear2.bias.data.copy_(torch.tensor([0, 1, 1], dtype=torch.float))
        elif self.stn_mode == 'translation_scale':
            self.fc_localization_global1.linear2.bias.data.copy_(torch.tensor([0, 0, 1, 1], dtype=torch.float))
            self.fc_localization_global2.linear2.bias.data.copy_(torch.tensor([0, 0, 1, 1], dtype=torch.float))
            self.fc_localization_local1.linear2.bias.data.copy_(torch.tensor([0, 0, 1, 1], dtype=torch.float))
            self.fc_localization_local2.linear2.bias.data.copy_(torch.tensor([0, 0, 1, 1], dtype=torch.float))
        elif self.stn_mode == 'rotation_translation':
            self.fc_localization_global1.linear2.bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))
            self.fc_localization_global2.linear2.bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))
            self.fc_localization_local1.linear2.bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))
            self.fc_localization_local2.linear2.bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))
        elif self.stn_mode == 'rotation_translation_scale':
            self.fc_localization_global1.linear2.bias.data.copy_(torch.tensor([0, 0, 0, 1, 1], dtype=torch.float))
            self.fc_localization_global2.linear2.bias.data.copy_(torch.tensor([0, 0, 0, 1, 1], dtype=torch.float))
            self.fc_localization_local1.linear2.bias.data.copy_(torch.tensor([0, 0, 0, 1, 1], dtype=torch.float))
            self.fc_localization_local2.linear2.bias.data.copy_(torch.tensor([0, 0, 0, 1, 1], dtype=torch.float))
            
    def _get_stn_mode_theta(self, theta, x):
        if self.stn_mode == 'affine':
            theta_new = theta.view(-1, 2, 3)
        else:
            theta_new = torch.zeros([x.size(0), 2, 3], dtype=torch.float32, device=x.get_device(), requires_grad=True)
            theta_new = theta_new + 0
            theta_new[:, 0, 0] = 1.0
            theta_new[:, 1, 1] = 1.0
            if self.stn_mode == 'translation':
                theta_new[:, 0, 2] = theta[:, 0]
                theta_new[:, 1, 2] = theta[:, 1]
            elif self.stn_mode == 'rotation':
                angle = theta[:, 0]
                theta_new[:, 0, 0] = torch.cos(angle)
                theta_new[:, 0, 1] = -torch.sin(angle)
                theta_new[:, 1, 0] = torch.sin(angle)
                theta_new[:, 1, 1] = torch.cos(angle)
            elif self.stn_mode == 'scale':
                theta_new[:, 0, 0] = theta[:, 0]
                theta_new[:, 1, 1] = theta[:, 1]
            elif self.stn_mode == 'shear':
                theta_new[:, 0, 1] = theta[:, 0]
                theta_new[:, 1, 0] = theta[:, 1]
            elif self.stn_mode == 'rotation_scale':
                angle = theta[:, 0]
                theta_new[:, 0, 0] = torch.cos(angle) * theta[:, 1]
                theta_new[:, 0, 1] = -torch.sin(angle)
                theta_new[:, 1, 0] = torch.sin(angle)
                theta_new[:, 1, 1] = torch.cos(angle) * theta[:, 2]
            elif self.stn_mode == 'translation_scale':
                theta_new[:, 0, 2] = theta[:, 0]
                theta_new[:, 1, 2] = theta[:, 1]
                theta_new[:, 0, 0] = theta[:, 2]
                theta_new[:, 1, 1] = theta[:, 3]
            elif self.stn_mode == 'rotation_translation':
                angle = theta[:, 0]
                theta_new[:, 0, 0] = torch.cos(angle)
                theta_new[:, 0, 1] = -torch.sin(angle)
                theta_new[:, 1, 0] = torch.sin(angle)
                theta_new[:, 1, 1] = torch.cos(angle)
                theta_new[:, 0, 2] = theta[:, 1]
                theta_new[:, 1, 2] = theta[:, 2]
            elif self.stn_mode == 'rotation_translation_scale':
                angle = theta[:, 0]
                theta_new[:, 0, 0] = torch.cos(angle) * theta[:, 3]
                theta_new[:, 0, 1] = -torch.sin(angle)
                theta_new[:, 1, 0] = torch.sin(angle)
                theta_new[:, 1, 1] = torch.cos(angle) * theta[:, 4]
                theta_new[:, 0, 2] = theta[:, 1]
                theta_new[:, 1, 2] = theta[:, 2]
        
        return theta_new
    
    def forward(self, x, invert_rpn_gradients):
        if invert_rpn_gradients:
            x = grad_reverse(x)
            
        if self.separate_localization_net:
            x_loc_features_g1 = self.localization_net_g1(x, invert_rpn_gradients)
            x_loc_features_g2 = self.localization_net_g2(x, invert_rpn_gradients)
            x_loc_features_l1 = self.localization_net_l1(x, invert_rpn_gradients)
            x_loc_features_l2 = self.localization_net_l2(x, invert_rpn_gradients)
    
            theta_g1 = self.fc_localization_global1(x_loc_features_g1, invert_rpn_gradients)
            theta_g2 = self.fc_localization_global2(x_loc_features_g2, invert_rpn_gradients)
            theta_l1 = self.fc_localization_local1(x_loc_features_l1, invert_rpn_gradients)
            theta_l2 = self.fc_localization_local2(x_loc_features_l2, invert_rpn_gradients)
            
            theta_g1 = self._get_stn_mode_theta(theta_g1, x_loc_features_g1)
            theta_g2 = self._get_stn_mode_theta(theta_g2, x_loc_features_g2)
            theta_l1 = self._get_stn_mode_theta(theta_l1, x_loc_features_l1)
            theta_l2 = self._get_stn_mode_theta(theta_l2, x_loc_features_l2)
            
        else:
            x_loc_features = self.localization_net(x, invert_rpn_gradients)
    
            theta_g1 = self.fc_localization_global1(x_loc_features, invert_rpn_gradients)
            theta_g2 = self.fc_localization_global2(x_loc_features, invert_rpn_gradients)
            theta_l1 = self.fc_localization_local1(x_loc_features, invert_rpn_gradients)
            theta_l2 = self.fc_localization_local2(x_loc_features, invert_rpn_gradients)
        
            theta_g1 = self._get_stn_mode_theta(theta_g1, x_loc_features)
            theta_g2 = self._get_stn_mode_theta(theta_g2, x_loc_features)
            theta_l1 = self._get_stn_mode_theta(theta_l1, x_loc_features)
            theta_l2 = self._get_stn_mode_theta(theta_l2, x_loc_features)
        
        self.affine_matrix_g1 = theta_g1.cpu().detach().numpy()
        self.affine_matrix_g2 = theta_g2.cpu().detach().numpy()
        self.affine_matrix_l1 = theta_l1.cpu().detach().numpy()
        self.affine_matrix_l2 = theta_l2.cpu().detach().numpy()
        # print(f"theta g1: {theta_g1}")
        # print(f"theta g2: {theta_g2}")
        # print(f"theta l1: {theta_l1}")
        # print(f"theta l2: {theta_l2}")
        
        gridg1 = F.affine_grid(theta_g1, size=list(x.size()[:2]) + [224, 224])
        g1 = F.grid_sample(x, gridg1)

        gridg2 = F.affine_grid(theta_g2, size=list(x.size()[:2]) + [224, 224])
        g2 = F.grid_sample(x, gridg2)

        gridl1 = F.affine_grid(theta_l1, size=list(x.size()[:2]) + [96, 96])
        l1 = F.grid_sample(x, gridl1)

        gridl2 = F.affine_grid(theta_l2, size=list(x.size()[:2]) + [96, 96])
        l2 = F.grid_sample(x, gridl2)
        
        return [g1, g2, l1, l2]
        
    
class AugmentationNetwork(nn.Module):
    def __init__(self, transform_net):
        super().__init__()
        print("Initializing Augmentation Network")
        self.transform_net = transform_net

    def forward(self, imgs, invert_rpn_gradients):
        global_views1_augmented, global_views2_augmented, local_views1_augmented, local_views2_augmented = [], [], [], []
        
        # since we have list of images with varying resolution, we need to transform them individually
        for img in imgs:
            img = torch.unsqueeze(img, 0)
            try:
                if img.size(2) > 700 or img.size(3) > 700:
                        img = resize(img, size=700, max_size=701)
            except Exception as e:
                print(e)
        
            if self.transform_net.invert_rpn_gradients:
                img = grad_reverse(img)
        
            global_local_views = self.transform_net(img, invert_rpn_gradients)
            g1_augmented = torch.squeeze(global_local_views[0], 0)
            g2_augmented = torch.squeeze(global_local_views[1], 0)
            l1_augmented = torch.squeeze(global_local_views[2], 0)
            l2_augmented = torch.squeeze(global_local_views[3], 0)

            global_views1_augmented.append(g1_augmented)
            global_views2_augmented.append(g2_augmented)
            local_views1_augmented.append(l1_augmented)
            local_views2_augmented.append(l2_augmented)
            
        global_views1 = torch.stack(global_views1_augmented, 0)
        global_views2 = torch.stack(global_views2_augmented, 0)
        local_views1 = torch.stack(local_views1_augmented, 0)
        local_views2 = torch.stack(local_views2_augmented, 0)

        del global_views1_augmented
        del global_views2_augmented
        del local_views1_augmented
        del local_views2_augmented

        return [global_views1, global_views2, local_views1, local_views2]
    