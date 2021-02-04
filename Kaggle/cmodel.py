import torch
import torch.nn as nn
import torch.nn.functional as F

class CModel(nn.Module):
    def __init__(self, num_classes):
        super(CModel, self).__init__()
        self.module_list = nn.ModuleList([CBlock(3, 8),
                                          CBlock((3**1)*8, (3**1)*8),
                                          CBlock((3**2)*8, (3**2)*8),
                                          CBlock((3**3)*8, (3**3)*8),
                                          CBlock((3**4)*8, (3**4)*8)])
        self.classifier = nn.Sequential(torch.nn.Linear((3**5)*8, 64, bias=True),
                                        torch.nn.PReLU(),
                                        torch.nn.Dropout(0.3),
                                        torch.nn.Linear(64, num_classes, bias=True),
                                        torch.nn.Sigmoid())

    def forward(self, x):
        x = self.module_list[0](x)
        x = self.module_list[1](x)
        x = self.module_list[2](x)
        x = self.module_list[3](x)
        x = self.module_list[4](x)
        x = self.classifier(x)
        return x

class CBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBlock, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.branch3x3 = BasicConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch5x5 = BasicConv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.branch_out_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_out = torch.cat([branch1x1, branch3x3, branch5x5], dim=1)
        branch_out_pool = self.branch_out_pool(branch_out)
        return branch_out_pool

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        #self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    def forward(self, x):
        x = self.conv(x)
        #x = self.bn(x)
        return F.relu(x, inplace=True)

cmodel = CModel(2)
cmodel(torch.Tensor(10,3,128,128).uniform_())