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
        return GradientReverse.scale * grad_output.neg()


def grad_reverse(x, scale=1.0):
    GradientReverse.scale = scale
    return GradientReverse.apply(x)

def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


class ResNetRPN(nn.Module):
    def __init__(self, backbone='resnet18', out_dim=256, invert_rpn_gradients=False, weights="ResNet18_Weights.IMAGENET1K_V1"):
        super().__init__()
        
        self.invert_rpn_gradients = invert_rpn_gradients
        
        if backbone == 'resnet18':
            backbone = resnet18(weights=weights)
            # summary(backbone.cuda(), (3, 224, 224))
        elif backbone == 'resnet34':
            backbone = resnet34(weights=weights)
        elif backbone == 'resnet50':
            backbone = resnet50(weights=weights)
        elif backbone == 'resnet101':
            backbone = resnet101(weights=weights)
        else:  # backbone == 'resnet152':
            backbone = resnet152(weights=weights)
        # if backbone_path:
        #     backbone.load_state_dict(torch.load(backbone_path))

        # backbone.fc = nn.Linear(512, out_dim)
        # torch.nn.init.xavier_uniform_(backbone.fc.weight)
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, x):
        if self.invert_rpn_gradients:
            x = grad_reverse(x)
            
        x = self.backbone(x)
        return x
    
    
class LocalizationNet(nn.Module):
    def __init__(self, invert_gradients, conv1_depth=16, conv2_depth=8):
        super().__init__()
        
        self.invert_gradients = invert_gradients
        self.conv2d_1 = nn.Conv2d(3, conv1_depth, kernel_size=3, padding=2)
        self.maxpool2d = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = nn.Conv2d(conv1_depth, conv2_depth, kernel_size=3, padding=2)
        self.avgpool = nn.AdaptiveAvgPool2d((16, 16))
        self.relu = nn.ReLU(True)
        
    def forward(self, x):
        if self.invert_gradients:
            x = grad_reverse(x)
            
        x = self.maxpool2d(F.relu(self.conv2d_1(x)))
        x = self.avgpool(F.relu(self.conv2d_2(x)))
        return x


class LocHead(nn.Module):
    def __init__(self, invert_gradients, stn_mode):
        super().__init__()
        
        self.invert_gradients = invert_gradients
        self.stn_n_params = N_PARAMS[stn_mode]
        
        self.linear1 = nn.Linear(512, 32)
        self.linear2 = nn.Linear(32, self.stn_n_params)
    
    def forward(self, x):
        if self.invert_gradients:
            x = grad_reverse(x)
        
        # x = torch.flatten(x, 1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    

class STN(nn.Module):
    """"
    Spatial Transformer Network with a ResNet localization backbone
    """""
    def __init__(self, stn_mode='affine', invert_rpn_gradients=False):
        super(STN, self).__init__()
        self.stn_mode = stn_mode
        self.stn_n_params = N_PARAMS[stn_mode]
        self.invert_rpn_gradients = invert_rpn_gradients

        self.activation = {}
        # Spatial transformer localization-network
        self.localization_net = ResNetRPN(backbone="resnet18", out_dim=256, invert_rpn_gradients=invert_rpn_gradients)

        # print(self.localization_net)
        self.localization_net.register_forward_hook(hook=get_activation('avgpool', activation=self.activation))  # output is detached

        # Regressors for the 3 * 2 affine matrix
        self.fc_localization_global1 = LocHead(invert_gradients=invert_rpn_gradients, stn_mode=stn_mode)
        self.fc_localization_global2 = LocHead(invert_gradients=invert_rpn_gradients, stn_mode=stn_mode)
        self.fc_localization_local1 = LocHead(invert_gradients=invert_rpn_gradients, stn_mode=stn_mode)
        self.fc_localization_local2 = LocHead(invert_gradients=invert_rpn_gradients, stn_mode=stn_mode)
        
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
    
    def forward(self, imgs):
        feature_maps = []
        with torch.no_grad():
            for img in imgs:
                img = torch.unsqueeze(img, 0)
                try:
                    if img.size(2) > 800 or img.size(3) > 800:
                            img = resize(img, size=800, max_size=801)
                except Exception as e:
                    print(e)
    
                # if self.invert_rpn_gradients:
                #     img = grad_reverse(img)
    
                self.localization_net(img)
                feature_map = torch.squeeze(self.activation['avgpool'], 0)
                feature_maps.append(feature_map)
    
            loc_img_features = torch.stack(feature_maps, 0)
        
        # if self.invert_rpn_gradients:
        #     loc_img_features = grad_reverse(loc_img_features)

        theta_g1 = self.fc_localization_global1(loc_img_features)
        theta_g2 = self.fc_localization_global2(loc_img_features)
        theta_l1 = self.fc_localization_local1(loc_img_features)
        theta_l2 = self.fc_localization_local2(loc_img_features)
    
        theta_g1 = self._get_stn_mode_theta(theta_g1, loc_img_features)
        theta_g2 = self._get_stn_mode_theta(theta_g2, loc_img_features)
        theta_l1 = self._get_stn_mode_theta(theta_l1, loc_img_features)
        theta_l2 = self._get_stn_mode_theta(theta_l2, loc_img_features)
        # print(f"theta g1: {theta_g1}")
        # print(f"theta g2: {theta_g2}")
        # print(f"theta l1: {theta_l1}")
        # print(f"theta l2: {theta_l2}")

        global_views1_augmented, global_views2_augmented, local_views1_augmented, local_views2_augmented = [], [], [], []

        gridg1 = F.affine_grid(theta_g1, size=[len(imgs), 3, 224, 224])
        gridg2 = F.affine_grid(theta_g2, size=[len(imgs), 3, 224, 224])
        gridl1 = F.affine_grid(theta_l1, size=[len(imgs), 3, 96, 96])
        gridl2 = F.affine_grid(theta_l2, size=[len(imgs), 3, 96, 96])
        
        for i, orig_image in enumerate(imgs):
            orig_image = torch.unsqueeze(orig_image, 0)
            grid_g1 = torch.unsqueeze(gridg1[i], 0)
            grid_g2 = torch.unsqueeze(gridg2[i], 0)
            grid_l1 = torch.unsqueeze(gridl1[i], 0)
            grid_l2 = torch.unsqueeze(gridl2[i], 0)
            
            g1 = F.grid_sample(orig_image, grid_g1)
            g2 = F.grid_sample(orig_image, grid_g2)
            l1 = F.grid_sample(orig_image, grid_l1)
            l2 = F.grid_sample(orig_image, grid_l2)

            global_views1_augmented.append(torch.squeeze(g1, 0))
            global_views2_augmented.append(torch.squeeze(g2, 0))
            local_views1_augmented.append(torch.squeeze(l1, 0))
            local_views2_augmented.append(torch.squeeze(l2, 0))
        
        return [torch.stack(global_views1_augmented, 0), torch.stack(global_views2_augmented, 0), torch.stack(local_views2_augmented, 0), torch.stack(local_views2_augmented, 0)]
        
 
class AugmentationNetwork(nn.Module):
    def __init__(self, stn_mode, invert_rpn_gradients):
        super().__init__()
        print("Initializing Augmentation Network")

        self.stn = STN(stn_mode, invert_rpn_gradients)
        
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        
    def forward(self, imgs):
        global_local_views = self.stn(imgs)
        return global_local_views
