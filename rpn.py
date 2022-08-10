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
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.transforms.functional import crop
from torchvision import transforms
import utils
import kornia


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
    def __init__(self, backbone='resnet50', backbone_path=None):
        super().__init__()
        if backbone == 'resnet18':
            backbone = resnet18(pretrained=not backbone_path)
        elif backbone == 'resnet34':
            backbone = resnet34(pretrained=not backbone_path)
        elif backbone == 'resnet50':
            backbone = resnet50(pretrained=not backbone_path)
        elif backbone == 'resnet101':
            backbone = resnet101(pretrained=not backbone_path)
        else:  # backbone == 'resnet152':
            backbone = resnet152(pretrained=not backbone_path)
        if backbone_path:
            backbone.load_state_dict(torch.load(backbone_path))

        backbone.fc = nn.Linear(512, 256)
        torch.nn.init.xavier_uniform_(backbone.fc.weight)
        self.backbone = backbone

    def forward(self, x):
        x = grad_reverse(x)
        x = self.backbone(x)
        return x


class RPN(nn.Module):
    def __init__(self, backbone=ResNetRPN('resnet18')):
        super().__init__()
        print("Initializing RPN")
        self.backbone = backbone
        self.global1_fc = nn.Linear(256, 2)
        self.global2_fc = nn.Linear(256, 2)
        self.local1_fc = nn.Linear(256, 2)
        self.local2_fc = nn.Linear(256, 2)
        
        print(self.backbone.named_parameters())
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    
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
        g_views1_cropped_transf, g_views2_cropped_transf, l_views1_cropped_transf, l_views2_cropped_transf = [], [], [], []

        # since we have list of images with varying resolution, we need to transform them individually
        # additionally, transforms.Compose still does not support processing batches :(
        for img in imgs:
            img = torch.unsqueeze(img, 0)
            emb = self.backbone(img)
            emb = grad_reverse(emb)
            g_view1 = self.global1_fc(emb)
            g_view2 = self.global2_fc(emb)
            l_view1 = self.local1_fc(emb)
            l_view2 = self.local1_fc(emb)

            img = torch.squeeze(img, 0)
            g_view1_cropped, g_view2_cropped, l_view1_cropped, l_view2_cropped = self._get_cropped_imgs(g_view1, g_view2, l_view1, l_view2, img)
            g_views1_cropped_transf.append(g_view1_cropped)
            g_views2_cropped_transf.append(g_view2_cropped)
            l_views1_cropped_transf.append(l_view1_cropped)
            l_views2_cropped_transf.append(l_view2_cropped)

        # since images now have same resolution, we can transform them batch-wise
        # g_view1_tensors = torch.stack(g_views1_cropped_batch, 0).cuda()
        # g_view2_tensors = torch.stack(g_views2_cropped_batch, 0).cuda()
        # l_view1_tensors = torch.stack(l_views1_cropped_batch, 0).cuda()
        # l_view2_tensors = torch.stack(l_views2_cropped_batch, 0).cuda()

        # g_view1_transf = self.modules_g1(g_view1_tensors)
        # g_view2_transf = self.modules_g2(g_view2_tensors)
        # l_view1_transf = self.modules_l(l_view1_tensors)
        # l_view2_transf = self.modules_l(l_view2_tensors)
        
        g_view1_transf = torch.stack(g_views1_cropped_transf, 0).cuda()
        g_view2_transf = torch.stack(g_views2_cropped_transf, 0).cuda()
        l_view1_transf = torch.stack(l_views1_cropped_transf, 0).cuda()
        l_view2_transf = torch.stack(l_views2_cropped_transf, 0).cuda()
        
        return [g_view1_transf, g_view2_transf, l_view1_transf, l_view2_transf]

    def _get_cropped_imgs(self, g_view1_coords, g_view2_cords, l_view1_coords, l_view2_coords, img):
        
        # using crop functionality with padding
        g_view1 = crop(img, top=g_view1_coords[:, 0].int(), left=g_view1_coords[:, 1].int(), height=244, width=224)
        g_view1 = self.modules_g1(g_view1)
    
        g_view2 = crop(img, top=g_view2_cords[:, 0].int(), left=g_view2_cords[:, 1].int(), height=244, width=224)
        g_view2 = self.modules_g2(g_view2)
        
        l_view1 = crop(img, top=l_view1_coords[:, 0].int(), left=l_view1_coords[:, 1].int(), height=96, width=96)
        l_view1 = self.modules_l(l_view1)

        l_view2 = crop(img, top=l_view2_coords[:, 0].int(), left=l_view2_coords[:, 1].int(), height=96, width=96)
        l_view2 = self.modules_l(l_view2)

        return g_view1, g_view2, l_view1, l_view2