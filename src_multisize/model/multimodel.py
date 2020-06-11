import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import models
import pretrainedmodels
from torch import nn
import random



random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)

class FCViewer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

    
'''Dual Path Networks in PyTorch.'''
class Bottleneck(nn.Module):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer):
        super(Bottleneck, self).__init__()
        self.out_planes = out_planes
        self.dense_depth = dense_depth

        self.conv1 = nn.Conv2d(last_planes, in_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.conv3 = nn.Conv2d(in_planes, out_planes+dense_depth, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes+dense_depth)

        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv2d(last_planes, out_planes+dense_depth, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes+dense_depth)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        x = self.shortcut(x)
        d = self.out_planes
        out = torch.cat([x[:,:d,:,:]+out[:,:d,:,:], x[:,d:,:,:], out[:,d:,:,:]], 1)
        out = F.relu(out)
        return out


class DPN(nn.Module):
    def __init__(self, cfg, num_classes):
        super(DPN, self).__init__()
        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) #
        self.bn1 = nn.BatchNorm2d(64)
        self.last_planes = 64
        self.layer1 = self._make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1)
        self.layer2 = self._make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        self.layer3 = self._make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=2)
        self.layer4 = self._make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2)
        self.linear = nn.Linear(out_planes[3]+(num_blocks[3]+1)*dense_depth[3], num_classes)

    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i,stride in enumerate(strides):
            layers.append(Bottleneck(self.last_planes, in_planes, out_planes, dense_depth, stride, i==0))
            self.last_planes = out_planes + (i+2) * dense_depth
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        # out = F.avg_pool2d(out, 3)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



def DPN26(num_classes=54):
    cfg = {
        'in_planes': (96,192,384,768),
        'out_planes': (256,512,1024,2048),
        'num_blocks': (2,2,2,2),
        'dense_depth': (16,32,24,128)
    }
    return DPN(cfg, num_classes=num_classes)

def DPN92(num_classes=54):
    cfg = {
        'in_planes': (96,192,384,768),
        'out_planes': (256,512,1024,2048),
        'num_blocks': (3,4,20,3),
        'dense_depth': (16,32,24,128)
    }
    return DPN(cfg, num_classes=num_classes)


class MultiModalNet(nn.Module):
    def __init__(self, backbone1, backbone2, drop, num_classes=54, pretrained=True):
        super().__init__()
        if pretrained:
            img_model = pretrainedmodels.__dict__[backbone1](num_classes=1000, pretrained='imagenet') #seresnext101
        else:
            img_model = pretrainedmodels.__dict__[backbone1](num_classes=1000, pretrained=None)

        self.num_classes = num_classes
        self.visit_model = DPN26()
        
        self.img_encoder = list(img_model.children())[:-2]
        self.img_encoder.append(nn.AdaptiveAvgPool2d(1))
        self.img_encoder = nn.Sequential(*self.img_encoder)

        if drop > 0:
            self.img_fc = nn.Sequential(FCViewer(),
                                    nn.Dropout(drop),
                                    nn.Linear(img_model.last_linear.in_features, 256))
                                    
        else:
            self.img_fc = nn.Sequential(
                FCViewer(),
                nn.Linear(img_model.last_linear.in_features, 256)
            )

        self.cls = nn.Linear(320, self.num_classes)

    def forward(self, x_img,x_vis):
        x_img = self.img_encoder(x_img)
        x_img = self.img_fc(x_img)
        x_vis=self.visit_model(x_vis)
        x_cat = torch.cat((x_img,x_vis),1)
        x_cat = self.cls(x_cat)
        return x_cat
