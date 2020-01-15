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
        self.net_2 = nn.Linear(512,1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):  ##x[0] is dark x[1] is light
        ####forward for dark####
        x_d = x[1]
        N, V, C, H, W = x_d.size ()
        x_d = x_d.view(-1, C, H, W)
        y_d = self.net2(x_d)
        y_d = y_d.view((int(x_d.shape[0]/self.num_views),self.num_views,y_d.shape[-3],y_d.shape[-2],y_d.shape[-1]))#(1, 21, 512, 1, 1)
        y_d = self.net_2 (torch.max (y_d, 1)[0].view (y_d.shape[0], -1))
        y_d = self.dropout(y_d)
        ####forward for dark####

        ####forward for light####
        x_l = x[0]
        N, V, C, H, W = x_l.size ()
        x_l = x_l.view (-1, C, H, W)
        y_l = self.net2 (x_l)
        y_l = y_l.view ((int (x_d.shape[0] / self.num_views), self.num_views, y_l.shape[-3], y_l.shape[-2],
                         y_l.shape[-1]))  # (1, 21, 512, 1, 1)
        y_l = self.net_2 (torch.max (y_l, 1)[0].view (y_l.shape[0], -1))
        y_l = self.dropout (y_l)
        y = self.sigmoid (y_l-y_d)
        ####forward for light####
        return y

class ConvColumn6 (nn.Module):

    def __init__(self, numclass):
        super (ConvColumn6, self).__init__ ()
        # self.img = img
        self.conv_layer1 = self._make_conv_layer (3, 32)
        self.conv_layer2 = self._make_conv_layer (32, 64)
        self.conv_layer3 = self._make_conv_layer (64, 124)
        self.conv_layer4 = self._make_conv_layer (124, 256)
        # self.conv_layer4 = nn.Conv3d (124, 256, kernel_size=(2, 3, 1), padding=0)
        self.conv_layer5 = nn.Conv3d (256, 512, kernel_size=(1, 3, 3), padding=0)
        self.sigmoid = nn.Sigmoid()
        self.fc5 = nn.Linear (512, 512)
        self.relu = nn.LeakyReLU ()
        self.batch0 = nn.BatchNorm1d (512)
        self.drop = nn.Dropout (p=0.15)
        self.fc6 = nn.Linear (512, 256)
        self.relu = nn.LeakyReLU ()
        self.batch1 = nn.BatchNorm1d (256)
        self.maxpool3d = nn.MaxPool3d((3, 3, 1))
        self.drop = nn.Dropout (p=0.15)
        self.fc7 = nn.Linear (256, numclass)

    def _make_conv_layer(self, in_c, out_c):
        conv_layer = nn.Sequential (
            nn.Conv3d (in_c, out_c, kernel_size=(2, 3, 3), padding=0),
            nn.BatchNorm3d (out_c),
            nn.LeakyReLU (),
            nn.Conv3d (out_c, out_c, kernel_size=(2, 3, 3), padding=1),
            nn.BatchNorm3d (out_c),
            nn.LeakyReLU (),
            nn.MaxPool3d ((2, 2, 2)),
        )
        return conv_layer

    def forward(self, x):
        # x=self.img
        print("before 1", x.size())
        x = self.conv_layer1 (x)
        print("after 1", x.size())
        x = self.conv_layer2 (x)
        print("after 2",x.size())
        x = self.conv_layer3 (x)
        print("after 3",x.size())
        x = self.conv_layer4 (x)
        print("after 4", x.size())
        x = self.conv_layer5 (x)
        print("after 5",x.size())
        x = self.maxpool3d(x)
        print ("after pool", x.size ())
        x = x.view (x.size (0), -1)
        x = self.fc5 (x)
        x = self.relu (x)
        print ("before batch0", x.size ())
        x = self.batch0 (x)
        x = self.drop (x)
        x = self.fc6 (x)
        x = self.relu (x)
        x = self.batch1 (x)
        x = self.drop (x)
        # x1 = x
        x = self.fc7 (x)
        x=self.sigmoid(x)
        return x