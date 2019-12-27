import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
import torchvision.models as models

class my_mvcnn(nn.Module):
    def __init__(self, num_views):
        super(my_mvcnn, self).__init__()
        self.net = models.resnet18(pretrained=True)
        self.net2 = nn.Sequential(*list(self.net.children())[:-1])
        self.num_views = num_views
        self.net_2 = nn.Linear(512,2)
    def forward(self, x):
        print("x>>>>>>>",x.size())
        y = self.net2(x)
        print("y>>>>>>", y.size())
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))#(1, 21, 512, 1, 1)
        print(y.size())
        y = self.net_2 (torch.max (y, 1)[0].view (y.shape[0], -1))
        print (y.size())
        return y