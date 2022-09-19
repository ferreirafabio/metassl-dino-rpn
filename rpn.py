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


class ResNetRPN(nn.Module):
    def __init__(self, backbone='resnet18', out_dim=512, invert_rpn_gradients=False):
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

    def forward(self, x):
        if self.invert_rpn_gradients:
            x = grad_reverse(x)
            
        x = self.backbone(x)
        return x
    
    
class LocalizationNet(nn.Module):
    def __init__(self, invert_gradients):
        super().__init__()
        
        self.invert_gradients = invert_gradients
        self.conv2d_1 = nn.Conv2d(1, 8, kernel_size=7)
        self.maxpool2d_1 = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = nn.Conv2d(8, 10, kernel_size=5)
        self.maxpool2d_2 = nn.MaxPool2d(2, stride=2)
        
    def forward(self, x):
        if self.invert_gradients:
            x = grad_reverse(x)
            
        x = F.relu(self.maxpool2d_1(self.conv2d_1(x)))
        x = F.relu(self.maxpool2d_2(self.conv2d_2(x)))
        print(x.size())
        return x


class LocHead(nn.Module):
    def __init__(self, invert_gradients, stn_mode):
        super().__init__()
        
        self.invert_gradients = invert_gradients
        self.stn_n_params = N_PARAMS[stn_mode]
        
        self.linear1 = nn.Linear(90, 32)
        self.linear2 = nn.Linear(32, self.stn_n_params)
    
    def forward(self, x):
        if self.invert_gradients:
            x = grad_reverse(x)
        
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    

class STN(nn.Module):
    """"
    Spatial Transformer Network with a ResNet localization backbone
    """""
    def __init__(self, backbone="resnet18", stn_mode='affine', localization_dim=512, invert_rpn_gradients=False):
        super(STN, self).__init__()
        self.stn_mode = stn_mode
        self.stn_n_params = N_PARAMS[stn_mode]
        self.localization_dim = localization_dim
        self.invert_rpn_gradients = invert_rpn_gradients
        
        # Spatial transformer localization-network
        # self.localization_net = ResNetRPN(backbone=backbone, out_dim=localization_dim, invert_rpn_gradients=invert_rpn_gradients)
        self.localization_net = LocalizationNet(invert_rpn_gradients)

        # Regressors for the 3 * 2 affine matrix
        self.fc_localization_global1 = LocHead(invert_gradients=invert_rpn_gradients, stn_mode=stn_mode)
        self.fc_localization_global2 = LocHead(invert_gradients=invert_rpn_gradients, stn_mode=stn_mode)
        self.fc_localization_local1 = LocHead(invert_gradients=invert_rpn_gradients, stn_mode=stn_mode)
        self.fc_localization_local2 = LocHead(invert_gradients=invert_rpn_gradients, stn_mode=stn_mode)
        
        # self.fc_localization_global1 = nn.Sequential(
        #     nn.Linear(self.localization_dim, 32),
        #     nn.ReLU(True),
        #     nn.Linear(32, self.stn_n_params)
        #     )

        # self.fc_localization_global2 = nn.Sequential(
        #     nn.Linear(self.localization_dim, 32),
        #     nn.ReLU(True),
        #     nn.Linear(32, self.stn_n_params)
        #     )

        # self.fc_localization_local1 = nn.Sequential(
        #     nn.Linear(self.localization_dim, 32),
        #     nn.ReLU(True),
        #     nn.Linear(32, self.stn_n_params)
        #     )
        #
        # self.fc_localization_local2 = nn.Sequential(
        #     nn.Linear(self.localization_dim, 32),
        #     nn.ReLU(True),
        #     nn.Linear(32, self.stn_n_params)
        #     )
        
        # Initialize the weights/bias with identity transformation
        # self.fc_localization_global1[2].weight.data.fill_(0)
        self.fc_localization_global1.linear2.weight.data.fill_(0)
        # self.fc_localization_global1[2].weight.data.zero_()
        self.fc_localization_global1.linear2.weight.data.zero_(0)
        
        # self.fc_localization_global2[2].weight.data.fill_(0)
        self.fc_localization_global2.linear2.weight.data.fill_(0)
        # self.fc_localization_global2[2].weight.data.zero_()
        self.fc_localization_global2.linear2.weight.data.zero_(0)
        
        # self.fc_localization_local1[2].weight.data.fill_(0)
        self.fc_localization_local1.linear2.weight.data.fill_(0)
        # self.fc_localization_local1[2].weight.data.zero_()
        self.fc_localization_local1.linear2.weight.data.zero_(0)
        
        # self.fc_localization_local2[2].weight.data.fill_(0)
        self.fc_localization_local2.linear2.weight.data.fill_(0)
        # self.fc_localization_local2[2].weight.data.zero_()
        self.fc_localization_local2.linear2.weight.data.zero_(0)
        
        if self.stn_mode == 'affine':
            # self.fc_localization_global1[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
            self.fc_localization_global1.linear2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
            # self.fc_localization_global2[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
            self.fc_localization_global2.linear2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
            # self.fc_localization_local1[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
            self.fc_localization_local1.linear2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
            # self.fc_localization_local2[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
            self.fc_localization_local2.linear2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
            
        elif self.stn_mode in ['translation', 'shear']:
            self.fc_localization_global1[2].bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
            self.fc_localization_global2[2].bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
            self.fc_localization_local1[2].bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
            self.fc_localization_local2[2].bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
        elif self.stn_mode == 'scale':
            self.fc_localization_global1[2].bias.data.copy_(torch.tensor([1, 1], dtype=torch.float))
            self.fc_localization_global2[2].bias.data.copy_(torch.tensor([1, 1], dtype=torch.float))
            self.fc_localization_local1[2].bias.data.copy_(torch.tensor([1, 1], dtype=torch.float))
            self.fc_localization_local2[2].bias.data.copy_(torch.tensor([1, 1], dtype=torch.float))
        elif self.stn_mode == 'rotation':
            self.fc_localization_global1[2].bias.data.copy_(torch.tensor([0], dtype=torch.float))
            self.fc_localization_global2[2].bias.data.copy_(torch.tensor([0], dtype=torch.float))
            self.fc_localization_local1[2].bias.data.copy_(torch.tensor([0], dtype=torch.float))
            self.fc_localization_local2[2].bias.data.copy_(torch.tensor([0], dtype=torch.float))
        elif self.stn_mode == 'rotation_scale':
            self.fc_localization_global1[2].bias.data.copy_(torch.tensor([0, 1, 1], dtype=torch.float))
            self.fc_localization_global2[2].bias.data.copy_(torch.tensor([0, 1, 1], dtype=torch.float))
            self.fc_localization_local1[2].bias.data.copy_(torch.tensor([0, 1, 1], dtype=torch.float))
            self.fc_localization_local2[2].bias.data.copy_(torch.tensor([0, 1, 1], dtype=torch.float))
        elif self.stn_mode == 'translation_scale':
            self.fc_localization_global1[2].bias.data.copy_(torch.tensor([0, 0, 1, 1], dtype=torch.float))
            self.fc_localization_global2[2].bias.data.copy_(torch.tensor([0, 0, 1, 1], dtype=torch.float))
            self.fc_localization_local1[2].bias.data.copy_(torch.tensor([0, 0, 1, 1], dtype=torch.float))
            self.fc_localization_local2[2].bias.data.copy_(torch.tensor([0, 0, 1, 1], dtype=torch.float))
        elif self.stn_mode == 'rotation_translation':
            self.fc_localization_global1[2].bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))
            self.fc_localization_global2[2].bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))
            self.fc_localization_local1[2].bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))
            self.fc_localization_local2[2].bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))
        elif self.stn_mode == 'rotation_translation_scale':
            self.fc_localization_global1[2].bias.data.copy_(torch.tensor([0, 0, 0, 1, 1], dtype=torch.float))
            self.fc_localization_global2[2].bias.data.copy_(torch.tensor([0, 0, 0, 1, 1], dtype=torch.float))
            self.fc_localization_local1[2].bias.data.copy_(torch.tensor([0, 0, 0, 1, 1], dtype=torch.float))
            self.fc_localization_local2[2].bias.data.copy_(torch.tensor([0, 0, 0, 1, 1], dtype=torch.float))
            
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
    
    def forward(self, x):
        if self.invert_rpn_gradients:
            x = grad_reverse(x)
            
        xs = self.localization_net(x)
        
        theta_g1 = self.fc_localization_global1(xs)
        theta_g2 = self.fc_localization_global2(xs)
        theta_l1 = self.fc_localization_local1(xs)
        theta_l2 = self.fc_localization_local2(xs)
        
        theta_g1 = self._get_stn_mode_theta(theta_g1, xs)
        theta_g2 = self._get_stn_mode_theta(theta_g2, xs)
        theta_l1 = self._get_stn_mode_theta(theta_l1, xs)
        theta_l2 = self._get_stn_mode_theta(theta_l2, xs)
        # print(f"theta g1: {theta_g1}")
        # print(f"theta g2: {theta_g2}")
        print(f"theta l1: {theta_l1}")
        print(f"theta l2: {theta_l2}")
        print(list(x.size()[:2]) + [224, 224])
        grid = F.affine_grid(theta_g1, size=list(x.size()[:2]) + [224, 224])
        g1 = F.grid_sample(x, grid)

        grid = F.affine_grid(theta_g2, size=list(x.size()[:2]) + [224, 224])
        g2 = F.grid_sample(x, grid)

        grid = F.affine_grid(theta_l1, size=list(x.size()[:2]) + [96, 96])
        l1 = F.grid_sample(x, grid)

        grid = F.affine_grid(theta_l2, size=list(x.size()[:2]) + [96, 96])
        l2 = F.grid_sample(x, grid)
        
        return [g1, g2, l1, l2]
        
    
class AugmentationNetwork(nn.Module):
    def __init__(self, transform_net):
        super().__init__()
        print("Initializing Augmentation Network")
        self.transform_net = transform_net

        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

        self.modules_g1 = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                utils.GaussianBlur(1.0),
                # kornia.augmentation.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
                self.normalize,
                ]
            )

        self.modules_g2 = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                utils.GaussianBlur(1.0),
                # kornia.augmentation.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
                utils.Solarization(0.2),
                # transforms.RandomSolarize(threshold=128, p=0.2),
                # kornia.augmentation.RandomSolarize(p=0.2),
                self.normalize,
                ]
            )

        self.modules_l = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                utils.GaussianBlur(0.5),
                # transforms.RandomApply(transforms.GaussianBlur(kernel_size=5), p=0.5),
                # kornia.augmentation.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5),
                self.normalize,
                ]
            )

    def forward(self, imgs):
        global_views1_augmented, global_views2_augmented, local_views1_augmented, local_views2_augmented = [], [], [], []
        
        # since we have list of images with varying resolution, we need to transform them individually
        for img in imgs:
            img = torch.unsqueeze(img, 0)
            try:
                if img.size(2) > 800 or img.size(3) > 800:
                        img = resize(img, size=800, max_size=801)
            except Exception as e:
                print(e)
        
            if self.transform_net.invert_rpn_gradients:
                img = grad_reverse(img)
        
            global_local_views = self.transform_net(img)
            g1_augmented = torch.squeeze(global_local_views[0], 0)
            g2_augmented = torch.squeeze(global_local_views[1], 0)
            l1_augmented = torch.squeeze(global_local_views[2], 0)
            l2_augmented = torch.squeeze(global_local_views[3], 0)
            # g1_augmented = self.modules_g1(torch.squeeze(global_local_views[0], 0))
            # g2_augmented = self.modules_g2(torch.squeeze(global_local_views[1], 0))
            # l1_augmented = self.modules_l(torch.squeeze(global_local_views[2], 0))
            # l2_augmented = self.modules_l(torch.squeeze(global_local_views[3], 0))

            global_views1_augmented.append(g1_augmented)
            global_views2_augmented.append(g2_augmented)
            local_views1_augmented.append(l1_augmented)
            local_views2_augmented.append(l2_augmented)
            
        # global_views1 = torch.stack(global_views1_augmented, 0).cuda()
        # global_views2 = torch.stack(global_views2_augmented, 0).cuda()
        # local_views1 = torch.stack(local_views1_augmented, 0).cuda()
        # local_views2 = torch.stack(local_views2_augmented, 0).cuda()

        global_views1 = torch.stack(global_views1_augmented, 0)
        global_views2 = torch.stack(global_views2_augmented, 0)
        local_views1 = torch.stack(local_views1_augmented, 0)
        local_views2 = torch.stack(local_views2_augmented, 0)

        del global_views1_augmented
        del global_views2_augmented
        del local_views1_augmented
        del local_views2_augmented

        return [global_views1, global_views2, local_views1, local_views2]
    