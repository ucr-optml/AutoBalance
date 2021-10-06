'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,BN=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        if BN:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
        else:
            self.bn1 = nn.Sequential()
            self.bn2 = nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if BN:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,in_planes=16,BN=True):
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        if BN:
            self.bn1 = nn.BatchNorm2d(in_planes)
        else:
            self.bn1 = nn.Sequential()
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1,BN=BN)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks[1], stride=2,BN=BN)
        self.layer3 = self._make_layer(block, in_planes*4, num_blocks[2], stride=2,BN=BN)
        self.avg_pool=nn.AvgPool2d(8)
        self.linear = nn.Linear(in_planes*4, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride,BN):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,BN=BN))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.layer4(out)
        out=self.avg_pool(out)
        feature = out.view(out.size(0), -1)
        logits = self.linear(feature)
        return logits

def ResNet20(num_classes=10,BN=True):
    num_res_blocks = int((20 - 2) / 6)
    return ResNet(BasicBlock, [num_res_blocks, num_res_blocks, num_res_blocks],num_classes=num_classes,BN=BN)

def ResNet32(num_classes=10,BN=True):
    num_res_blocks = int((32 - 2) / 6)
    return ResNet(BasicBlock, [num_res_blocks, num_res_blocks, num_res_blocks],num_classes=num_classes,BN=BN)

def ResNet44(num_classes=10,BN=True):
    num_res_blocks = int((44 - 2) / 6)
    return ResNet(BasicBlock, [num_res_blocks, num_res_blocks, num_res_blocks],num_classes=num_classes,BN=BN)

def ResNet56(num_classes=10,BN=True):
    num_res_blocks = int((56 - 2) / 6)
    return ResNet(BasicBlock, [num_res_blocks, num_res_blocks, num_res_blocks],num_classes=num_classes,BN=BN)

def ResNet110(num_classes=10,BN=True):
    num_res_blocks = int((110 - 2) / 6)
    return ResNet(BasicBlock, [num_res_blocks, num_res_blocks, num_res_blocks],num_classes=num_classes,BN=BN)

def ResNet_n(layer,num_classes=10,BN=True):
    num_res_blocks = int((layer - 2) / 6)
    return ResNet(BasicBlock, [num_res_blocks, num_res_blocks, num_res_blocks],num_classes=num_classes,BN=BN)

def test():
    net = ResNet20(BN=True)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y)
    print(y[0].size(),y[1].size())

#test()