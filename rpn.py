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
                utils.Solarization(0.2),
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
                self.normalize,
                ]
            )
        
    def forward(self, imgs):
        # g_view1_tensors, g_view2_tensors, l_view1_tensors, l_view2_tensors = [], [], [], []
        g_views1_cropped_batch, g_views2_cropped_batch, l_views1_cropped_batch, l_views2_cropped_batch = [], [], [], []

        # since we have list of images with varying resolution, we need to transform them individually
        # additionally, transforms.Compose still does not support processing batches :(
        for img in imgs:
            emb = self.backbone(img)
            g_view1 = self.global1_fc(emb)
            g_view2 = self.global2_fc(emb)
            l_view1 = self.local1_fc(emb)
            l_view2 = self.local1_fc(emb)
        
            g_view1_cropped, g_view2_cropped, l_view1_cropped, l_view2_cropped = self._get_cropped_imgs(g_view1, g_view2, l_view1, l_view2, img)
            g_views1_cropped_batch.append(g_view1_cropped)
            g_views2_cropped_batch.append(g_view2_cropped)
            l_views1_cropped_batch.append(l_view1_cropped)
            l_views2_cropped_batch.append(l_view2_cropped)

        print("test")
        # since images now have same resolution, we can transform them batch-wise
        g_view1_tensors = torch.stack(g_views1_cropped_batch, 0).cuda()
        g_view2_tensors = torch.stack(g_views2_cropped_batch, 0).cuda()
        l_view1_tensors = torch.stack(l_views1_cropped_batch, 0).cuda()
        l_view2_tensors = torch.stack(l_views2_cropped_batch, 0).cuda()
        print(l_view2_tensors.size())

        g_view1_transf = self.modules_g1(g_view1_tensors)
        g_view2_transf = self.modules_g2(g_view2_tensors)
        l_view1_transf = self.modules_l(l_view1_tensors)
        l_view2_transf = self.modules_l(l_view2_tensors)
        
        return [g_view1_transf, g_view2_transf, l_view1_transf, l_view2_transf]
        
        # for img in imgs:
        #     img = torch.unsqueeze(img, 0)
        #     emb = self.backbone(img)
        #     g_view1 = self.global1_fc(emb)
        #     g_view2 = self.global2_fc(emb)
        #     l_view1 = self.local1_fc(emb)
        #     l_view2 = self.local1_fc(emb)
            
            # img = torch.squeeze(img, 0)
            # g_view1, g_view2, l_view1, l_view2 = self._get_cropped_imgs(g_view1, g_view2, l_view1, l_view2, img)
            # views.append(self._get_cropped_imgs(g_view1, g_view2, l_view1, l_view2, img))
            # g_view1_tensors.append(g_view1)
            # g_view2_tensors.append(g_view2)
            # l_view1_tensors.append(l_view1)
            # l_view2_tensors.append(l_view2)
    
        # g_view1_tensors = torch.stack(g_view1_tensors, 0).cuda()
        # g_view2_tensors = torch.stack(g_view2_tensors, 0).cuda()
        # l_view1_tensors = torch.stack(l_view1_tensors, 0).cuda()
        # l_view2_tensors = torch.stack(l_view2_tensors, 0).cuda()
        
        # print("-------------------------")
        # print(g_view1_tensors.size())
        # print(g_view2_tensors.size())
        # print(l_view1_tensors.size())
        # print(l_view2_tensors.size())

        # return [g_view1_tensors, g_view2_tensors, l_view1_tensors, l_view2_tensors]

    def _get_cropped_imgs(self, g_view1_coords, g_view2_cords, l_view1_coords, l_view2_coords, img):
        
        # using crop functionality with padding
        g_view1 = crop(img, top=g_view1_coords[:, 0].int(), left=g_view1_coords[:, 1].int(), height=244, width=224)
        # g_view1 = self.modules_g1(g_view1)
    
        g_view2 = crop(img, top=g_view2_cords[:, 0].int(), left=g_view2_cords[:, 1].int(), height=244, width=224)
        # g_view2 = self.modules_g2(g_view2)
        
        l_view1 = crop(img, top=l_view1_coords[:, 0].int(), left=l_view1_coords[:, 1].int(), height=96, width=96)
        # l_view1 = self.modules_l(l_view1)

        l_view2 = crop(img, top=l_view2_coords[:, 0].int(), left=l_view2_coords[:, 1].int(), height=96, width=96)
        # l_view2 = self.modules_l(l_view2)

        return g_view1, g_view2, l_view1, l_view2
        
        
class SSD300(nn.Module):
    def __init__(self, backbone=ResNetRPN('resnet50')):
        super().__init__()

        self.feature_extractor = backbone

        self.label_num = 81  # number of COCO classes
        self._build_additional_features(self.feature_extractor.out_channels)
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        self.loc = []
        self.conf = []

        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            self.loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            self.conf.append(nn.Conv2d(oc, nd * self.label_num, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        self._init_weights()

    def _build_additional_features(self, input_size):
        self.additional_blocks = []
        for i, (input_size, output_size, channels) in enumerate(zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])):
            if i < 3:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )

            self.additional_blocks.append(layer)

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def _init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1: nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def bbox_view(self, src, loc, conf):
        ret = []
        for s, l, c in zip(src, loc, conf):
            ret.append((l(s).view(s.size(0), 4, -1), c(s).view(s.size(0), self.label_num, -1)))

        locs, confs = list(zip(*ret))
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, x):
        x = self.feature_extractor(x)

        detection_feed = [x]
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)

        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)

        # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
        return locs, confs

