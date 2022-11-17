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
from torchvision.transforms.functional import crop, resize
from torchvision import transforms
import utils
import kornia
from torchvision.models import resnet

# from torchsummary import summary


# needed for the spatial transformer net
N_PARAMS = {
        'affine': 6,
        'translation': 2,
        'rotation': 1,
        'scale': 2,
        'scale_symmetric': 1,
        'shear': 2,
        'rotation_scale': 3,
        'translation_scale': 4,
        'rotation_translation': 3,
        'rotation_translation_scale': 5,
        'rotation_scale_symmetric': 2,
        'rotation_translation_scale_symmetric_limited': 4,
        'rotation_translation_scale_symmetric_limited_0_1': 4,
        'rotation_translation_scale_symmetric_limited_10': 4,
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

    
class LocalizationNet(nn.Module):
    def __init__(self, conv1_depth=32, conv2_depth=32, deep=False, invert_rpn_gradients=True):
        super().__init__()
        
        self.deep = deep
        self.invert_rpn_gradients = invert_rpn_gradients
        self.conv2d_1 = nn.Conv2d(3, conv1_depth, kernel_size=3, padding=2)
        self.conv2d_bn1 = nn.BatchNorm2d(conv1_depth)
        self.maxpool2d = nn.MaxPool2d(2, stride=2)
        
        if self.deep:
            self.conv2d_deep1 = nn.Conv2d(conv1_depth, conv1_depth, kernel_size=3, padding=2)
            self.conv2d_deep_bn1 = nn.BatchNorm2d(conv1_depth)
            
            self.conv2d_deep2 = nn.Conv2d(conv1_depth, conv1_depth, kernel_size=3, padding=2)
            self.conv2d_deep_bn2 = nn.BatchNorm2d(conv1_depth)
            
        self.conv2d_2 = nn.Conv2d(conv1_depth, conv2_depth, kernel_size=3, padding=2)
        self.conv2d_bn2 = nn.BatchNorm2d(conv2_depth)
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        
    def forward(self, x):
    
        xs = self.maxpool2d(F.leaky_relu(self.conv2d_bn1(self.conv2d_1(x))))

        if self.deep:
            xs = self.maxpool2d(F.leaky_relu(self.conv2d_deep_bn1(self.conv2d_deep1(xs))))
            xs = self.maxpool2d(F.leaky_relu(self.conv2d_deep_bn2(self.conv2d_deep2(xs))))

        if self.invert_rpn_gradients:
            xs = grad_reverse(self.avgpool(F.leaky_relu(self.conv2d_bn2(self.conv2d_2(xs)))))
        else:
            xs = self.avgpool(F.leaky_relu(self.conv2d_bn2(self.conv2d_2(xs))))
            
        return xs


class LocHead(nn.Module):
    def __init__(self, stn_mode, conv2_depth, invert_rpn_gradients=True, use_unbounded_stn=False):
        super().__init__()
        
        self.stn_n_params = N_PARAMS[stn_mode]
        self.invert_rpn_gradients = invert_rpn_gradients
        self.use_unbounded_stn = use_unbounded_stn
    
        self.linear0 = nn.Linear(8 * 8 * conv2_depth, 128)
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, self.stn_n_params)
    
    def forward(self, x):

        xs = torch.flatten(x, 1)
        xs = F.leaky_relu(self.linear0(xs))
        xs = F.leaky_relu(self.linear1(xs))

        if self.invert_rpn_gradients:
            xs = grad_reverse(self.linear2(xs))
        else:
            xs = self.linear2(xs)
            
        return xs
    

class STN(nn.Module):
    """"
    Spatial Transformer Network with a ResNet localization backbone
    """""
    def __init__(self, stn_mode='affine',
                 separate_localization_net=False,
                 invert_rpn_gradients=True,
                 deep_loc_net=False,
                 use_one_res=False,
                 use_unbounded_stn=False,
                 conv1_depth=32,
                 conv2_depth=32,
                 ):
        super(STN, self).__init__()
        self.stn_mode = stn_mode
        self.stn_n_params = N_PARAMS[stn_mode]
        self.separate_localization_net = separate_localization_net
        self.invert_rpn_gradients = invert_rpn_gradients
        self.deep_loc_net = deep_loc_net
        self.use_one_res = use_one_res
        self.use_unbounded_stn = use_unbounded_stn
        self.conv1_depth = conv1_depth
        self.conv2_depth = conv2_depth
        
        self.affine_matrix_g1 = None
        self.affine_matrix_g2 = None
        self.affine_matrix_l1 = None
        self.affine_matrix_l2 = None
        
        # Spatial transformer localization-network
        if self.separate_localization_net:
            self.conv1_depth = 8
            self.conv2_depth = 16
            print(f"Using separate localization networks. Overriding conv1_depth={self.conv1_depth} and conv2_depth={self.conv2_depth}.")
            self.localization_net_g1 = LocalizationNet(conv1_depth=conv1_depth, conv2_depth=conv2_depth, deep=False, invert_rpn_gradients=self.invert_rpn_gradients)
            self.localization_net_g2 = LocalizationNet(conv1_depth=conv1_depth, conv2_depth=conv2_depth, deep=False, invert_rpn_gradients=self.invert_rpn_gradients)
            self.localization_net_l1 = LocalizationNet(conv1_depth=conv1_depth, conv2_depth=conv2_depth, deep=False, invert_rpn_gradients=self.invert_rpn_gradients)
            self.localization_net_l2 = LocalizationNet(conv1_depth=conv1_depth, conv2_depth=conv2_depth, deep=False, invert_rpn_gradients=self.invert_rpn_gradients)
        else:
            if self.deep_loc_net:
                self.localization_net = LocalizationNet(conv1_depth=self.conv1_depth,
                                                        conv2_depth=self.conv2_depth,
                                                        deep=True,
                                                        invert_rpn_gradients=self.invert_rpn_gradients,
                                                        )
            else:
                self.localization_net = LocalizationNet(conv1_depth=self.conv1_depth,
                                                        conv2_depth=self.conv2_depth,
                                                        deep=False,
                                                        invert_rpn_gradients=self.invert_rpn_gradients,
                                                        )

        # Regressors for the 3 * 2 affine matrix
        self.fc_localization_global1 = LocHead(stn_mode=stn_mode,
                                               conv2_depth=conv2_depth,
                                               invert_rpn_gradients=self.invert_rpn_gradients,
                                               use_unbounded_stn=self.use_unbounded_stn,
                                               )
        self.fc_localization_global2 = LocHead(stn_mode=stn_mode,
                                               conv2_depth=conv2_depth,
                                               invert_rpn_gradients=self.invert_rpn_gradients,
                                               use_unbounded_stn=self.use_unbounded_stn,
                                               )
        self.fc_localization_local1 = LocHead(stn_mode=stn_mode,
                                              conv2_depth=conv2_depth,
                                              invert_rpn_gradients=self.invert_rpn_gradients,
                                              use_unbounded_stn=self.use_unbounded_stn,
                                              )
        self.fc_localization_local2 = LocHead(stn_mode=stn_mode,
                                              conv2_depth=conv2_depth,
                                              invert_rpn_gradients=self.invert_rpn_gradients,
                                              use_unbounded_stn=self.use_unbounded_stn,
                                              )
        
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
        elif self.stn_mode == 'scale_symmetric':
            self.fc_localization_global1.linear2.bias.data.copy_(torch.tensor([1], dtype=torch.float))
            self.fc_localization_global2.linear2.bias.data.copy_(torch.tensor([1], dtype=torch.float))
            self.fc_localization_local1.linear2.bias.data.copy_(torch.tensor([1], dtype=torch.float))
            self.fc_localization_local2.linear2.bias.data.copy_(torch.tensor([1], dtype=torch.float))
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
        elif self.stn_mode == 'rotation_scale_symmetric':
            self.fc_localization_global1.linear2.bias.data.copy_(torch.tensor([0, 1], dtype=torch.float))
            self.fc_localization_global2.linear2.bias.data.copy_(torch.tensor([0, 1], dtype=torch.float))
            self.fc_localization_local1.linear2.bias.data.copy_(torch.tensor([0, 1], dtype=torch.float))
            self.fc_localization_local2.linear2.bias.data.copy_(torch.tensor([0, 1], dtype=torch.float))
        elif self.stn_mode == 'rotation_translation_scale_symmetric_limited':
            # a transformation that is symmetric in scale, i.e. s_x = s_y as well as limited (translation and scale in [-0.5, +0.5])
            self.fc_localization_global1.linear2.bias.data.copy_(torch.tensor([0, 0, 0, 1], dtype=torch.float))
            self.fc_localization_global2.linear2.bias.data.copy_(torch.tensor([0, 0, 0, 1], dtype=torch.float))
            self.fc_localization_local1.linear2.bias.data.copy_(torch.tensor([0, 0, 0, 1], dtype=torch.float))
            self.fc_localization_local2.linear2.bias.data.copy_(torch.tensor([0, 0, 0, 1], dtype=torch.float))
        elif self.stn_mode == 'rotation_translation_scale_symmetric_limited_0_1':
            # a transformation that is symmetric in scale, i.e. s_x = s_y as well as limited (translation and scale in [-0.1, +0.1])
            self.fc_localization_global1.linear2.bias.data.copy_(torch.tensor([0, 0, 0, 1], dtype=torch.float))
            self.fc_localization_global2.linear2.bias.data.copy_(torch.tensor([0, 0, 0, 1], dtype=torch.float))
            self.fc_localization_local1.linear2.bias.data.copy_(torch.tensor([0, 0, 0, 1], dtype=torch.float))
            self.fc_localization_local2.linear2.bias.data.copy_(torch.tensor([0, 0, 0, 1], dtype=torch.float))
        elif self.stn_mode == 'rotation_translation_scale_symmetric_limited_10':
            # a transformation that is symmetric in scale, i.e. s_x = s_y as well as limited (translation in [-0.1, +0.1], scale in [-10, +10])
            self.fc_localization_global1.linear2.bias.data.copy_(torch.tensor([0, 0, 0, 1], dtype=torch.float))
            self.fc_localization_global2.linear2.bias.data.copy_(torch.tensor([0, 0, 0, 1], dtype=torch.float))
            self.fc_localization_local1.linear2.bias.data.copy_(torch.tensor([0, 0, 0, 1], dtype=torch.float))
            self.fc_localization_local2.linear2.bias.data.copy_(torch.tensor([0, 0, 0, 1], dtype=torch.float))
        
    def _get_stn_mode_theta(self, theta, x):
        # print(theta.shape) # torch.Size([1, 6])
        if self.stn_mode == 'affine':
            theta[:, 1:] = theta[:, 1:] if self.use_unbounded_stn else torch.tanh(theta[:, 1:])  # optionally bound everything except for the angle at [:, 0]
            theta_new = theta.view(-1, 2, 3)
            # print(theta_new.shape) # torch.Size([1, 2, 3])
        else:
            theta_new = torch.zeros([x.size(0), 2, 3], dtype=torch.float32, device=x.get_device(), requires_grad=True)
            theta_new = theta_new + 0
            theta_new[:, 0, 0] = 1.0
            theta_new[:, 1, 1] = 1.0
            if self.stn_mode == 'translation':
                theta_new[:, 0, 2] = theta[:, 0] if self.use_unbounded_stn else torch.tanh(theta[:, 0])
                theta_new[:, 1, 2] = theta[:, 1] if self.use_unbounded_stn else torch.tanh(theta[:, 1])
            elif self.stn_mode == 'rotation':
                angle = theta[:, 0]  # leave unbounded<
                theta_new[:, 0, 0] = torch.cos(angle)
                theta_new[:, 0, 1] = -torch.sin(angle)
                theta_new[:, 1, 0] = torch.sin(angle)
                theta_new[:, 1, 1] = torch.cos(angle)
            elif self.stn_mode == 'scale':
                theta_new[:, 0, 0] = theta[:, 0] if self.use_unbounded_stn else torch.tanh(theta[:, 0])
                theta_new[:, 1, 1] = theta[:, 1] if self.use_unbounded_stn else torch.tanh(theta[:, 1])
            elif self.stn_mode == 'scale_symmetric':
                theta_new[:, 0, 0] = theta[:, 0] if self.use_unbounded_stn else torch.tanh(theta[:, 0])
                theta_new[:, 1, 1] = theta[:, 0] if self.use_unbounded_stn else torch.tanh(theta[:, 0])
            elif self.stn_mode == 'shear':
                theta_new[:, 0, 1] = theta[:, 0] if self.use_unbounded_stn else torch.tanh(theta[:, 0])
                theta_new[:, 1, 0] = theta[:, 1] if self.use_unbounded_stn else torch.tanh(theta[:, 1])
            elif self.stn_mode == 'rotation_scale':
                angle = theta[:, 0]  # leave unbounded
                theta_new[:, 0, 0] = torch.cos(angle) * (theta[:, 1] if self.use_unbounded_stn else torch.tanh(theta[:, 1]))
                theta_new[:, 0, 1] = -torch.sin(angle)
                theta_new[:, 1, 0] = torch.sin(angle)
                theta_new[:, 1, 1] = torch.cos(angle) * (theta[:, 2] if self.use_unbounded_stn else torch.tanh(theta[:, 2]))
            elif self.stn_mode == 'translation_scale':
                theta_new[:, 0, 2] = theta[:, 0] if self.use_unbounded_stn else torch.tanh(theta[:, 0])
                theta_new[:, 1, 2] = theta[:, 1] if self.use_unbounded_stn else torch.tanh(theta[:, 1])
                theta_new[:, 0, 0] = theta[:, 2] if self.use_unbounded_stn else torch.tanh(theta[:, 2])
                theta_new[:, 1, 1] = theta[:, 3] if self.use_unbounded_stn else torch.tanh(theta[:, 3])
            elif self.stn_mode == 'rotation_translation':
                angle = theta[:, 0]  # leave unbounded
                theta_new[:, 0, 0] = torch.cos(angle)
                theta_new[:, 0, 1] = -torch.sin(angle)
                theta_new[:, 1, 0] = torch.sin(angle)
                theta_new[:, 1, 1] = torch.cos(angle)
                theta_new[:, 0, 2] = theta[:, 1] if self.use_unbounded_stn else torch.tanh(theta[:, 1])
                theta_new[:, 1, 2] = theta[:, 2] if self.use_unbounded_stn else torch.tanh(theta[:, 2])
            elif self.stn_mode == 'rotation_translation_scale':
                angle = theta[:, 0]  # leave unbounded
                theta_new[:, 0, 0] = torch.cos(angle) * (theta[:, 3] if self.use_unbounded_stn else torch.tanh(theta[:, 3]))
                theta_new[:, 0, 1] = -torch.sin(angle)
                theta_new[:, 1, 0] = torch.sin(angle)
                theta_new[:, 1, 1] = torch.cos(angle) * (theta[:, 4] if self.use_unbounded_stn else torch.tanh(theta[:, 4]))
                theta_new[:, 0, 2] = theta[:, 1] if self.use_unbounded_stn else torch.tanh(theta[:, 1])
                theta_new[:, 1, 2] = theta[:, 2] if self.use_unbounded_stn else torch.tanh(theta[:, 2])
            elif self.stn_mode == 'rotation_scale_symmetric':
                # rotation_scale sometimes leads to strong distortions along only one axis (x or y), this is used to make the scaling symmetric along both axes
                angle = theta[:, 0]  # leave unbounded
                theta_new[:, 0, 0] = torch.cos(angle) * (theta[:, 1] if self.use_unbounded_stn else torch.tanh(theta[:, 1]))
                theta_new[:, 0, 1] = -torch.sin(angle)
                theta_new[:, 1, 0] = torch.sin(angle)
                theta_new[:, 1, 1] = torch.cos(angle) * (theta[:, 1] if self.use_unbounded_stn else torch.tanh(theta[:, 1]))
            elif self.stn_mode == 'rotation_translation_scale_symmetric_limited':
                angle = theta[:, 0]  # leave unbounded
                theta_new[:, 0, 0] = torch.cos(angle) * (theta[:, 3] if self.use_unbounded_stn else torch.mul(torch.tanh(theta[:, 3]), .5))
                theta_new[:, 0, 1] = -torch.sin(angle)
                theta_new[:, 1, 0] = torch.sin(angle)
                theta_new[:, 1, 1] = torch.cos(angle) * (theta[:, 3] if self.use_unbounded_stn else torch.mul(torch.tanh(theta[:, 3]), .5))
                theta_new[:, 0, 2] = theta[:, 1] if self.use_unbounded_stn else torch.mul(torch.tanh(theta[:, 1]), .5)
                theta_new[:, 1, 2] = theta[:, 2] if self.use_unbounded_stn else torch.mul(torch.tanh(theta[:, 2]), .5)
            elif self.stn_mode == 'rotation_translation_scale_symmetric_limited_0_1':
                angle = theta[:, 0]  # leave unbounded
                theta_new[:, 0, 0] = torch.cos(angle) * (theta[:, 3] if self.use_unbounded_stn else torch.mul(torch.tanh(theta[:, 3]), .1))
                theta_new[:, 0, 1] = -torch.sin(angle)
                theta_new[:, 1, 0] = torch.sin(angle)
                theta_new[:, 1, 1] = torch.cos(angle) * (theta[:, 3] if self.use_unbounded_stn else torch.mul(torch.tanh(theta[:, 3]), .1))
                theta_new[:, 0, 2] = theta[:, 1] if self.use_unbounded_stn else torch.mul(torch.tanh(theta[:, 1]), .1)
                theta_new[:, 1, 2] = theta[:, 2] if self.use_unbounded_stn else torch.mul(torch.tanh(theta[:, 2]), .1)
            elif self.stn_mode == 'rotation_translation_scale_symmetric_limited_10':
                angle = theta[:, 0]  # leave unbounded
                theta_new[:, 0, 0] = torch.cos(angle) * (theta[:, 3] if self.use_unbounded_stn else torch.mul(torch.tanh(theta[:, 3]), 10))
                theta_new[:, 0, 1] = -torch.sin(angle)
                theta_new[:, 1, 0] = torch.sin(angle)
                theta_new[:, 1, 1] = torch.cos(angle) * (theta[:, 3] if self.use_unbounded_stn else torch.mul(torch.tanh(theta[:, 3]), 10))
                theta_new[:, 0, 2] = theta[:, 1] if self.use_unbounded_stn else torch.mul(torch.tanh(theta[:, 1]), .1)
                theta_new[:, 1, 2] = theta[:, 2] if self.use_unbounded_stn else torch.mul(torch.tanh(theta[:, 2]), .1)
        
        return theta_new
    
    def forward(self, x):
        if self.separate_localization_net:
            x_loc_features_g1 = self.localization_net_g1(x)
            x_loc_features_g2 = self.localization_net_g2(x)
            x_loc_features_l1 = self.localization_net_l1(x)
            x_loc_features_l2 = self.localization_net_l2(x)
    
            theta_g1 = self.fc_localization_global1(x_loc_features_g1)
            theta_g2 = self.fc_localization_global2(x_loc_features_g2)
            theta_l1 = self.fc_localization_local1(x_loc_features_l1)
            theta_l2 = self.fc_localization_local2(x_loc_features_l2)
            
            theta_g1 = self._get_stn_mode_theta(theta_g1, x_loc_features_g1)
            theta_g2 = self._get_stn_mode_theta(theta_g2, x_loc_features_g2)
            theta_l1 = self._get_stn_mode_theta(theta_l1, x_loc_features_l1)
            theta_l2 = self._get_stn_mode_theta(theta_l2, x_loc_features_l2)
            
        else:
            x_loc_features = self.localization_net(x)
    
            theta_g1 = self.fc_localization_global1(x_loc_features)
            theta_g2 = self.fc_localization_global2(x_loc_features)
            theta_l1 = self.fc_localization_local1(x_loc_features)
            theta_l2 = self.fc_localization_local2(x_loc_features)
        
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
        
        high_res = 224
        low_res = 96
        one_res = 128
        if self.use_one_res:
            low_res, high_res = one_res, one_res
        
        gridg1 = F.affine_grid(theta_g1, size=list(x.size()[:2]) + [high_res, high_res])
        g1 = F.grid_sample(x, gridg1)

        gridg2 = F.affine_grid(theta_g2, size=list(x.size()[:2]) + [high_res, high_res])
        g2 = F.grid_sample(x, gridg2)

        gridl1 = F.affine_grid(theta_l1, size=list(x.size()[:2]) + [low_res, low_res])
        l1 = F.grid_sample(x, gridl1)

        gridl2 = F.affine_grid(theta_l2, size=list(x.size()[:2]) + [low_res, low_res])
        l2 = F.grid_sample(x, gridl2)
        
        return [g1, g2, l1, l2]
        
    
class AugmentationNetwork(nn.Module):
    def __init__(self, transform_net):
        super().__init__()
        print("Initializing Augmentation Network")
        self.transform_net = transform_net

    def forward(self, imgs):
        global_views1_list, global_views2_list, local_views1_list, local_views2_list = [], [], [], []
        
        # since we have list of images with varying resolution, we need to transform them individually
        for img in imgs:
            img = torch.unsqueeze(img, 0)
            try:
                if img.size(2) > 700 or img.size(3) > 700:
                        img = resize(img, size=700, max_size=701)
            except Exception as e:
                print(e)
                
            global_local_views = self.transform_net(img)
            
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

        return [global_views1, global_views2, local_views1, local_views2]
    
