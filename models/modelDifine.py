import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from opt.opt import *
from torch.autograd import Variable
from models.box_filter import BoxFilter
import numpy as np

opt = NetOption()

def load_models(model, modelpath):
    if modelpath != None:
        print(">>>>>>>>>>load model")
        mydict = model.state_dict()
        state_dict = torch.load(modelpath)
        pretrained_dict = {k: v for k, v in state_dict.items() if k not in ["fc.bias", 'fc.weight']}
        mydict.update(pretrained_dict)
        model.load_state_dict(mydict)
    return model

class my_svcnn(nn.Module):
    def __init__(self, num_views):
        super(my_svcnn, self).__init__()
        # self.pretrainpath = pretrain
        self.net = load_models(models.resnet18(pretrained=True), self.pretrainpath)
        self.innet2 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.net2 = nn.Sequential(*list(self.net.children())[1:-1])
        self.num_views = num_views
        self.net_2 = nn.Linear(512,1)
        # torch.nn.init.xavier_uniform(self.net_2.weight)  ##xavier 初始化参数
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):  ##x[0] is dark x[1] is light
        x = self.innet2(x)
        x = self.net2 (x)
        x = self.net_2 (x)
        x = self.dropout (x)
        x = self.sigmoid (x)
        return x

class my_mvcnn(nn.Module):
    def __init__(self, num_views):
        super(my_mvcnn, self).__init__()
        # self.pretrainpath = pretrain
        self.net = models.resnet18(pretrained=False)
        self.innet2 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.net2 = nn.Sequential(*list(self.net.children())[1:-1])
        self.num_views = num_views
        self.net_2 = nn.Linear(512,1)
        # torch.nn.init.xavier_uniform(self.net_2.weight)  ##xavier 初始化参数
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        # self.net = load_models(my_mvcnn, self.pretrainpath)
        if opt.retrain:
            print("retrain model >>>>>>")
            retrain = torch.load(opt.retrain)["model"]
            if isinstance(retrain, nn.DataParallel):
                a = retrain
                self.innet2 = retrain.module.innet2
                # self.net2  =nn.Sequential(*list(retrain.module.net.children())[1:-1])
                self.bn1 = retrain.module.net.bn1
                self.relu = retrain.module.net.relu
                self.maxpool = retrain.module.net.maxpool
                self.layer1 = retrain.module.net.layer1
                self.layer2 = retrain.module.net.layer2
                self.layer3 = retrain.module.net.layer3
                self.layer4 = retrain.module.net.layer4
                self.avgpool = retrain.module.net.avgpool
                self.net2 = nn.Sequential(self.bn1,
                                          self.relu,
                                          self.maxpool,
                                          self.layer1,
                                          self.layer2,
                                          self.layer3,
                                          self.layer4,
                                          self.avgpool)
                self.net_2 = retrain.module.net_2
                # self.fc = retrain.module.fc
                # self.model = retrain.module.model
            else:
                a = retrain
                self.conv1 = retrain.conv1
                self.bn1 = retrain.bn1
                # self.relu = retrain.relu
                self.maxpool = retrain.maxpool
                self.layer1 = retrain.layer1
                self.layer2 = retrain.layer2
                self.layer3 = retrain.layer3
                self.layer4 = retrain.layer4
                self.avgpool = retrain.avgpool
                self.fc = retrain.fc
                # self.model = retrain.model
            return
    def forward(self, x):  ##x[0] is dark x[1] is light
        ####forward for dark####
        x_d = x[0]
        N, V, C, H, W = x_d.size ()
        x_d = x_d.view(-1, C, H, W)

        # y_d = self.net2(x_d)
        # print(y_d.size())
        # y_d = y_d.view((int(x_d.shape[0]/self.num_views),self.num_views,y_d.shape[-3],y_d.shape[-2],y_d.shape[-1]))#(1, 21, 512, 1, 1)
        # print(y_d.size())
        # y_d = self.net_2 (torch.max (y_d, 1)[0].view (y_d.shape[0], -1))
        # y_d = self.dropout(y_d)
        # y_d = self.sigmoid (y_d)
        ####forward for dark####

        ####forward for light####
        x_l = x[1]
        N, V, C, H, W = x_l.size ()
        x_l = x_l.view (-1, C, H, W)
        x_cat = torch.cat ((x_d, x_l), 1)
        xsize = x_cat.size()
        y_l = self.innet2(x_cat)
        # size1 = y_l.size()
        y_l = self.net2 (y_l)
        # ysize=y_l.size()
        y_l = y_l.view ((int (x_cat.shape[0] / self.num_views), self.num_views, y_l.shape[-3], y_l.shape[-2],
                         y_l.shape[-1]))  # (1, 21, 512, 1, 1)
        y_l = self.net_2 (torch.max (y_l, 1)[0].view (y_l.shape[0], -1))
        y_l = self.dropout (y_l)
        y_l = self.sigmoid (y_l)
        ####forward for light####
        return y_l

class my_mvcnn_lstm(nn.Module):
    def __init__(self, num_views):
        super(my_mvcnn_lstm, self).__init__()
        self.net = models.resnet18(pretrained=True)
        # self.innet2 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.net2 = nn.Sequential(*list(self.net.children())[0:-1])
        self.num_views = num_views
        self.lstm1 = LSTM(512, 21, 3, 1)
        self.lstm2 = LSTM(512, 21, 3, 1)
        #self.net_2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):  ##x[0] is dark x[1] is light
        ####forward for dark####
        x_d = x[1]
        N, V, C, H, W = x_d.size ()
        x_d = x_d.view(-1, C, H, W)
        x_l = x[0]
        N, V, C, H, W = x_l.size ()
        x_l = x_l.view (-1, C, H, W)
        # y_l = self.innet2(x_l)
        y_l = self.net2 (x_l)
        y_l = y_l.view ((int (x_l.shape[0] / self.num_views), self.num_views, y_l.shape[-3], y_l.shape[-2],
                         y_l.shape[-1]))  # (4, 21, 512, 1, 1)
        #print("y_l.shape1<<",y_l.size())
        # y_l = torch.squeeze(y_l)
        y_l = y_l.view((-1, y_l.shape[1], y_l.shape[2]))
        y_l = torch.transpose(y_l, 1, 0).contiguous()
        print("y_l.shape", y_l.shape)
        #print("y_i.shape2", y_l.size())
        # y_d = self.innet2(x_d)
        y_d = self.net2(x_d)
        y_d = y_d.view((int(x_d.shape[0] / self.num_views), self.num_views, y_d.shape[-3], y_d.shape[-2],
                        y_d.shape[-1]))
        # y_d = torch.squeeze(y_d)
        y_d = y_d.view((-1, y_d.shape[1], y_d.shape[2]))
        y_d = torch.transpose(y_d, 1, 0).contiguous()
        pred_d, out = self.lstm1(y_d)
        pred_l, _ = self.lstm2(y_l, out)
        pre = self.dropout(pred_l)
        pre = self.sigmoid(pre)
        #print("pre.size", pre.size())
        return pre

        #y_l = self.net_2 (torch.max (y_l, 1)[0].view(y_l.shape[0], -1))

        # y_l = self.dropout (y_l)
        # y_l = self.sigmoid (y_l)
        ####forward for light####
        #return y_l



class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(LSTM, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, n_class)
    def forward(self, x, hidden= None):
        pre, out = self.lstm(x, hidden)
        pre = pre[-1, :, :]
        pre = self.classifier(pre)
        return pre, out

class my_mvcnn_lstm1(nn.Module):
    def __init__(self, num_views):
        super(my_mvcnn_lstm1, self).__init__()
        self.net = models.resnet18(pretrained=True)
        # self.innet2 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.net2 = nn.Sequential(*list(self.net.children())[:-1])
        self.num_views = num_views
        self.lstm1 = LSTM(512, 512, 3, 1)
        self.lstm2 = LSTM(512, 512, 3, 1)
        #self.net_2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):  ##x[0] is dark x[1] is light
        ####forward for dark####
        x_d = x[1]
        N, V, C, H, W = x_d.size ()
        x_d = x_d.view(-1, C, H, W)
        x_l = x[0]
        N, V, C, H, W = x_l.size ()
        x_l = x_l.view (-1, C, H, W)
        y_l = self.innet2(x_l)
        y_l = self.net2 (y_l)
        y_l = y_l.view ((int (x_l.shape[0] / self.num_views), self.num_views, y_l.shape[-3], y_l.shape[-2],
                         y_l.shape[-1]))  # (4, 21, 512, 1, 1)
        # y_l = torch.squeeze(y_l)
        y_l = y_l.view((-1, y_l.shape[1], y_l.shape[2]))
        y_d = self.innet2(x_d)
        y_d = self.net2(y_d)
        y_d = y_d.view((int(x_d.shape[0] / self.num_views), self.num_views, y_d.shape[-3], y_d.shape[-2],
                        y_d.shape[-1]))
        # y_d = torch.squeeze(y_d)
        y_d = y_d.view((-1, y_d.shape[1], y_d.shape[2]))
        pred_d, out = self.lstm1(y_d)
        pred_l, _ = self.lstm2(y_l, out)
        pre = self.sigmoid(pred_l)
        return pre

        #y_l = self.net_2 (torch.max (y_l, 1)[0].view(y_l.shape[0], -1))

        # y_l = self.dropout (y_l)
        # y_l = self.sigmoid (y_l)
        ####forward for light####
        #return y_l



class LSTM1(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(LSTM1, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, n_class)
    def forward(self, x, hidden= None):
        pre, out = self.lstm(x, hidden)
        pre = pre[:, -1, :]
        pre = self.classifier(pre)
        return pre, out


############################################################
class NonLocalBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner):
        super(NonLocalBlock, self).__init__()

        self.dim_in = dim_in
        self.dim_inner = dim_inner
        self.dim_out = dim_out

        self.theta = nn.Conv3d(dim_in, dim_inner, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
        self.phi = nn.Conv3d(dim_in, dim_inner, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.g = nn.Conv3d(dim_in, dim_inner, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

        self.out = nn.Conv3d(dim_inner, dim_out, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.bn = nn.BatchNorm3d(dim_out)

    def forward(self, x):
        residual = x

        batch_size = x.shape[0]
        mp = self.maxpool(x)
        theta = self.theta(x)
        phi = self.phi(mp)
        g = self.g(mp)

        theta_shape_5d = theta.shape
        theta, phi, g = theta.view(batch_size, self.dim_inner, -1), phi.view(batch_size, self.dim_inner, -1), g.view(
            batch_size, self.dim_inner, -1)

        theta_phi = torch.bmm(theta.transpose(1, 2), phi)  # (8, 1024, 784) * (8, 1024, 784) => (8, 784, 784)
        theta_phi_sc = theta_phi * (self.dim_inner ** -.5)
        p = F.softmax(theta_phi_sc, dim=-1)

        t = torch.bmm(g, p.transpose(1, 2))
        t = t.view(theta_shape_5d)

        out = self.out(t)
        out = self.bn(out)

        out = out + residual
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride, downsample, temp_conv, temp_stride, use_nl=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1 + temp_conv * 2, 1, 1), stride=(temp_stride, 1, 1), padding=(temp_conv, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        outplanes = planes * 4
        self.nl = NonLocalBlock(outplanes, outplanes, outplanes//2) if use_nl else None


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        if self.nl is not None:
            out = self.nl(out)

        return out
#############################################


class resnet3d(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=400, use_nl=False):
        self.inplanes = 64
        super(resnet3d, self).__init__()

        # self.gf = FastGuidedFilter_attention(r=2, eps=0.01)
        #
        # # attention blocks
        # self.attentionblock5 = GridAttentionBlock(in_channels=1024)
        ###conv fusion
        # self.conv11_top = nn.Conv3d(6, 3, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        # self.conv11_m_top = nn.Conv3d(6, 3, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.conv11 = nn.Conv3d(2048, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        ###conv fusion
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(2, 2, 2), padding=(2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 0, 0))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        nonlocal_mod = 2 if use_nl else 1000
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, temp_conv=[1, 1, 1], temp_stride=[1, 1, 1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, temp_conv=[1, 0, 1, 0],
                                       temp_stride=[1, 1, 1, 1], nonlocal_mod=nonlocal_mod)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, temp_conv=[1, 0, 1, 0, 1, 0],
                                       temp_stride=[1, 1, 1, 1, 1, 1], nonlocal_mod=nonlocal_mod)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, temp_conv=[0, 1, 0], temp_stride=[1, 1, 1])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        n=1
        if opt.cat ==True:
            n=1
        if opt.contra ==True:
            n=2
        self.fc = nn.Linear(n*512 * block.expansion, num_classes)

        ##### multi scale
        self.conv11_m = nn.Conv3d(2048, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.conv11_final = nn.Conv3d(4096, 2048, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        ###conv fusion
        self.inplanes = 64
        self.conv1_m = nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(2, 2, 2), padding=(2, 3, 3), bias=False)
        self.bn1_m = nn.BatchNorm3d(64)
        self.relu_m = nn.ReLU(inplace=True)
        self.maxpool1_m = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 0, 0))
        self.maxpool2_m = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        nonlocal_mod = 2 if use_nl else 1000
        self.layer1_m = self._make_layer(block, 64, layers[0], stride=1, temp_conv=[1, 1, 1], temp_stride=[1, 1, 1])
        self.layer2_m = self._make_layer(block, 128, layers[1], stride=2, temp_conv=[1, 0, 1, 0],
                                       temp_stride=[1, 1, 1, 1], nonlocal_mod=nonlocal_mod)
        self.layer3_m = self._make_layer(block, 256, layers[2], stride=2, temp_conv=[1, 0, 1, 0, 1, 0],
                                       temp_stride=[1, 1, 1, 1, 1, 1], nonlocal_mod=nonlocal_mod)
        self.layer4_m = self._make_layer(block, 512, layers[3], stride=2, temp_conv=[0, 1, 0], temp_stride=[1, 1, 1])
        self.avgpool_m = nn.AdaptiveAvgPool3d((1, 1, 1))
        n = 1
        self.fc_m = nn.Linear(n * 512 * block.expansion, num_classes)

        self.fc_contra = nn.Sequential(
            nn.Linear(n * 512 * block.expansion, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

        #####
        self.drop = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.extract = ConvBlock(3,3)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride, temp_conv, temp_stride, nonlocal_mod=1000):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or temp_stride[0] != 1:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=(1, 1, 1),
                          stride=(temp_stride[0], stride, stride), padding=(0, 0, 0), bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, temp_conv[0], temp_stride[0], False))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, temp_conv[i], temp_stride[i],
                                i % nonlocal_mod == nonlocal_mod - 1))

        return nn.Sequential(*layers)

    def forward_single(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.drop(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

    def forward_single_mscale_single(self, x):
        x_d, x_l, fullx_d, fullx_l = x
        # x_d = self.conv1(x_d)
        # x_d = self.bn1(x_d)
        # x_d = self.relu(x_d)
        # x_d = self.maxpool1(x_d)
        # x_d = self.layer1(x_d)
        # x_d = self.maxpool2(x_d)
        # x_d = self.layer2(x_d)
        # x_d = self.layer3(x_d)

        x_l = self.conv1(x_l)
        x_l = self.bn1(x_l)
        x_l = self.relu(x_l)
        x_l = self.maxpool1(x_l)
        x_l = self.layer1(x_l)
        x_l = self.maxpool2(x_l)
        x_l = self.layer2(x_l)
        x_l = self.layer3(x_l)
        # x = torch.cat((x_l, x_d), dim=1)
        # x = self.conv11(x)
        # print("xsize layer3", x_l.size())
        x_l = self.layer4(x_l)
        # fullx_d = self.conv1_m(fullx_d)
        # fullx_d = self.bn1_m(fullx_d)
        # fullx_d = self.relu_m(fullx_d)
        # fullx_d = self.maxpool1_m(fullx_d)
        # fullx_d = self.layer1_m(fullx_d)
        # fullx_d = self.maxpool2_m(fullx_d)
        # fullx_d = self.layer2_m(fullx_d)
        # fullx_d = self.layer3_m(fullx_d)

        fullx_l = self.conv1_m(fullx_l)
        fullx_l = self.bn1_m(fullx_l)
        fullx_l = self.relu_m(fullx_l)
        fullx_l = self.maxpool1_m(fullx_l)
        fullx_l = self.layer1_m(fullx_l)
        fullx_l = self.maxpool2_m(fullx_l)
        fullx_l = self.layer2_m(fullx_l)
        fullx_l = self.layer3_m(fullx_l)
        #
        # fullx = torch.cat((fullx_l, fullx_d), dim=1)
        # fullx = self.conv11_m(fullx)
        fullx_l = self.layer4_m(fullx_l)

        x = torch.cat((fullx_l, x_l), dim=1)
        x = self.conv11_final(x)
        x = self.avgpool(x)
        x = self.drop(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x


    def forward_single_mscale(self, x):
        x_d, x_l, fullx_d, fullx_l = x
        x_d = self.conv1(x_d)
        x_d = self.bn1(x_d)
        x_d = self.relu(x_d)
        x_d = self.maxpool1(x_d)
        x_d = self.layer1(x_d)
        x_d = self.maxpool2(x_d)
        x_d = self.layer2(x_d)
        x_d = self.layer3(x_d)

        x_l = self.conv1(x_l)
        x_l = self.bn1(x_l)
        x_l = self.relu(x_l)
        x_l = self.maxpool1(x_l)
        x_l = self.layer1(x_l)
        x_l = self.maxpool2(x_l)
        x_l = self.layer2(x_l)
        x_l = self.layer3(x_l)
        x = torch.cat((x_l, x_d), dim=1)
        x = self.conv11(x)
        # print("xsize layer3", x_l.size())
        x = self.layer4(x)
        fullx_d = self.conv1_m(fullx_d)
        fullx_d = self.bn1_m(fullx_d)
        fullx_d = self.relu_m(fullx_d)
        fullx_d = self.maxpool1_m(fullx_d)
        fullx_d = self.layer1_m(fullx_d)
        fullx_d = self.maxpool2_m(fullx_d)
        fullx_d = self.layer2_m(fullx_d)
        fullx_d = self.layer3_m(fullx_d)

        fullx_l = self.conv1_m(fullx_l)
        fullx_l = self.bn1_m(fullx_l)
        fullx_l = self.relu_m(fullx_l)
        fullx_l = self.maxpool1_m(fullx_l)
        fullx_l = self.layer1_m(fullx_l)
        fullx_l = self.maxpool2_m(fullx_l)
        fullx_l = self.layer2_m(fullx_l)
        fullx_l = self.layer3_m(fullx_l)

        fullx = torch.cat((fullx_l, fullx_d), dim=1)
        fullx = self.conv11_m(fullx)
        fullx = self.layer4_m(fullx)

        x = torch.cat((fullx, x), dim=1)
        x = self.conv11_final(x)
        x = self.avgpool(x)
        x = self.drop(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x

    def forward_single_cat(self, x):
        x_d,x_l = x
        x_d = self.conv1(x_d)
        x_d = self.bn1(x_d)
        x_d = self.relu(x_d)
        x_d = self.maxpool1(x_d)

        x_d = self.layer1(x_d)
        x_d = self.maxpool2(x_d)
        x_d = self.layer2(x_d)
        x_d = self.layer3(x_d)
        # x_d = self.layer4(x_d)

        x_l = self.conv1(x_l)
        x_l = self.bn1(x_l)
        x_l = self.relu(x_l)
        x_l = self.maxpool1(x_l)
        x_l = self.layer1(x_l)
        x_l = self.maxpool2(x_l)
        x_l = self.layer2(x_l)
        x_l = self.layer3(x_l)
        x = torch.cat((x_l, x_d), dim=1)
        x = self.conv11(x)
        # print("xsize layer3", x_l.size())
        x = self.layer4(x)
        # x = torch.cat((x_l, x_d), dim=1)
        # print("xsize layer4",x_l.size())
        # x = self.conv11(x)
        # print("xsize 11", x.size())
        x = self.avgpool(x)
        x = self.drop(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

    def forward_single_contra(self, x):
        x_d, x_l = x
        # print(x_d.size())
        x_d = self.conv1(x_d)
        x_d = self.bn1(x_d)
        x_d = self.relu(x_d)
        x_d = self.maxpool1(x_d)

        x_d = self.layer1(x_d)
        x_d = self.maxpool2(x_d)
        x_d = self.layer2(x_d)
        x_d = self.layer3(x_d)
        x_d = self.layer4(x_d)
        x_d = self.avgpool(x_d)
        ###get similarity vector
        h_d = x_d
        h_d = h_d.view(h_d.size(0), -1)
        h_d = self.fc_contra(h_d)
        ###
        x_d = self.drop(x_d)
        # print(x_d.size())
        # x_d = x_d.view(x_d.shape[0], -1)
        # x_d = self.fc_d(x_d)

        x_l = self.conv1(x_l)
        x_l = self.bn1(x_l)
        x_l = self.relu(x_l)
        x_l = self.maxpool1(x_l)
        x_l = self.layer1(x_l)
        x_l = self.maxpool2(x_l)
        x_l = self.layer2(x_l)
        x_l = self.layer3(x_l)
        x_l = self.layer4(x_l)
        x_l = self.avgpool(x_l)
        ###get similarity vector
        h_l = x_l
        h_l = h_l.view(h_l.size(0), -1)
        sizehl = h_l.size()
        h_l = self.fc_contra(h_l)
        ###


        x = torch.cat((x_l,x_d),dim=1)
        x = self.drop(x)
        x = x.view(x.shape[0], -1)

        x = self.fc(x)
        x = self.sigmoid(x)
        return h_d, h_l, x

    def forward_multi(self, x):
        clip_preds = []
        for clip_idx in range(x.shape[1]):  # B, 10, 3, 3, 32, 224, 224
            spatial_crops = []
            for crop_idx in range(x.shape[2]):
                clip = x[:, clip_idx, crop_idx]
                clip = self.forward_single(clip)
                spatial_crops.append(clip)
            spatial_crops = torch.stack(spatial_crops, 1).mean(1)  # (B, 400)
            clip_preds.append(spatial_crops)
        clip_preds = torch.stack(clip_preds, 1).mean(1)  # (B, 400)
        return clip_preds

    def forward(self, batch):##x[0] is dark x[1] is light Pair = (dark_input_var, light_input_var, fulldark_var, fulllight_var)
        x_d = batch[0]
        x_l = batch[1]
        fullx_d = batch[2]
        fullx_l = batch[3]
        x = (x_d, x_l)
        ms_x = (x_d, x_l, fullx_d, fullx_l)
        batch = {'frames': x, 'frames1': ms_x} ##0 dark 1 light
        # 5D tensor == single clip
        if opt.contra == True:
            pred = self.forward_single_contra(batch['frames'])
        if opt.mscale == True:
            # print("multiscale cat")
            pred = self.forward_single_mscale_single(batch['frames1'])
            # pred = self.forward_single_mscale(batch['frames1'])
        if opt.cat == True:
            # if batch['frames'].dim() == 5:
            # print("catmodel")
            pred = self.forward_single_cat(batch['frames'])
        # if opt.cat == False:
        #     if opt.attention ==True:
        #         print("attention")
        #         pred = self.attention_forward_single(batch['frames'])
        #     else:
        #         print("light")
        #         pred = self.forward_single(batch['frames'][1])###0 denotes dark 1 denotes light

        # 7D tensor == 3 crops/10 clips
        # elif batch['frames'].dim() == 7:
        #     pred = self.forward_multi(batch['frames'])

        loss_dict = {}
        if 'label' in batch:
            loss = F.cross_entropy(pred, batch['label'], reduction='none')
            loss_dict = {'loss': loss}

        return pred#, loss_dict

def i3_res50_nl(num_classes):
    net = resnet3d(num_classes=num_classes, use_nl=True)
    # state_dict = torch.load('pretrained/i3d_r50_nl_kinetics.pth')
    # net.load_state_dict(state_dict)
    # freeze_bn(net, "net") # Only needed for finetuning. For validation, .eval() works.
    return net

class FastGuidedFilter_attention(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(FastGuidedFilter_attention, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)
        self.epss = 1e-12

    def forward(self, lr_x, lr_y, hr_x, l_a):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()
        # print(lr_x.shape)
        # print(lr_y.shape)
        # print(hr_x.shape)
        lr_x = lr_x.double()
        lr_y = lr_y.double()
        hr_x = hr_x.double()
        l_a = l_a.double()

        assert n_lrx == n_lry and n_lry == n_hrx
        # assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry or c_lry ==1)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2*self.r+1 and w_lrx > 2*self.r+1

        ## N
        N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        # l_a = torch.abs(l_a)
        l_a = torch.abs(l_a) + self.epss

        # --------------------------------
        # The previous calculation of l_a is wrong
        # changed by zhang shihao at 2019/3/21
        #
        # previous:
        t_all = torch.sum(l_a)
        l_t = l_a / t_all
        # --------------------------------
        # l_t = l_a / self.boxfilter(l_a)
        # l_t = self.boxfilter(l_a)

        ## mean_attention
        mean_a = self.boxfilter(l_a) / N
        ## mean_a^2xy
        mean_a2xy = self.boxfilter(l_a * l_a * lr_x * lr_y) / N
        ## mean_tax
        mean_tax = self.boxfilter(l_t * l_a * lr_x) / N
        ## mean_ay
        mean_ay = self.boxfilter(l_a * lr_y) / N
        ## mean_a^2x^2
        mean_a2x2 = self.boxfilter(l_a * l_a * lr_x * lr_x) / N
        ## mean_ax
        mean_ax = self.boxfilter(l_a * lr_x) / N

        temp = torch.abs(mean_a2x2 - N * mean_tax * mean_ax)

        ## A
        # A = (mean_a2xy - N * mean_tax * mean_ay) / (mean_a2x2 - N * mean_tax * mean_ax + self.eps)
        A = (mean_a2xy - N * mean_tax * mean_ay) / (temp + self.eps)
        ## b
        b = (mean_ay - A * mean_ax) / (mean_a)

        # --------------------------------
        # Mean
        # --------------------------------
        A = self.boxfilter(A) / N
        b = self.boxfilter(b) / N


        ## mean_A; mean_b
        mean_A = F.upsample(A, (h_hrx, w_hrx), mode='bilinear')
        mean_b = F.upsample(b, (h_hrx, w_hrx), mode='bilinear')

        return (mean_A*hr_x+mean_b).float()


class GridAttentionBlock(nn.Module):###attention 模块
    def __init__(self, in_channels):
        super(GridAttentionBlock, self).__init__()

        self.inter_channels = in_channels
        self.in_channels = in_channels
        self.gating_channels = in_channels

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1)

        self.phi = nn.Conv2d(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(self.in_channels),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode='bilinear')
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        return sigm_psi_f

class dual_resnet3d(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=400, use_nl=False):
        self.inplanes = 64
        super(dual_resnet3d, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(2, 2, 2), padding=(2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 0, 0))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        nonlocal_mod = 2 if use_nl else 1000
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, temp_conv=[1, 1, 1], temp_stride=[1, 1, 1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, temp_conv=[1, 0, 1, 0],
                                       temp_stride=[1, 1, 1, 1], nonlocal_mod=nonlocal_mod)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, temp_conv=[1, 0, 1, 0, 1, 0],
                                       temp_stride=[1, 1, 1, 1, 1, 1], nonlocal_mod=nonlocal_mod)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, temp_conv=[0, 1, 0], temp_stride=[1, 1, 1])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.drop = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if opt.pretrain:
            print("pretrain model >>>>>>")
            retrain = torch.load(opt.pretrain)["model"]
            if isinstance(retrain, nn.DataParallel):
                # a = retrain
                self.conv1 = retrain.module.conv1p
                self.bn1 = retrain.module.bn1
                self.relu = retrain.module.relu
                self.maxpool1 = retrain.module.maxpool1
                self.maxpool2 = retrain.module.maxpool2
                self.layer1 = retrain.module.layer1
                self.layer2 = retrain.module.layer2
                self.layer3_o = retrain.module.layer3
                self.layer4_o = retrain.module.layer4
                self.avgpool_o = retrain.module.avgpool
                self.fc_o = retrain.module.fc

                self.layer3_s = retrain.module.layer3
                self.layer4_s = retrain.module.layer4
                self.avgpool_s = retrain.module.avgpool
                self.fc_s = retrain.module.fc
                # self.model = retrain.module.model
            else:
                a = retrain
                self.conv1 = retrain.conv1
                self.bn1 = retrain.bn1
                self.relu = retrain.relu
                self.maxpool1 = retrain.maxpool1
                self.maxpool2 = retrain.maxpool2
                self.layer1 = retrain.layer1
                self.layer2 = retrain.layer2
                self.layer3 = retrain.layer3
                self.layer4 = retrain.layer4
                self.avgpool = retrain.avgpool
                self.fc = retrain.fc
                # self.model = retrain.model
        self.extract = ConvBlock(3,3)

    def _make_layer(self, block, planes, blocks, stride, temp_conv, temp_stride, nonlocal_mod=1000):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or temp_stride[0] != 1:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=(1, 1, 1),
                          stride=(temp_stride[0], stride, stride), padding=(0, 0, 0), bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, temp_conv[0], temp_stride[0], False))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, temp_conv[i], temp_stride[i],
                                i % nonlocal_mod == nonlocal_mod - 1))

        return nn.Sequential(*layers)

    def forward_single(self, x):
        # x = self.extract(x)
        # x_structure = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.maxpool2(x)
        x = self.layer2(x)
        x_o = self.layer3_o(x)
        x_o = self.layer4_o(x_o)

        x_o = self.avgpool_o(x_o)  ##x_o for opennarrow   x_s for synechia
        x_o = self.drop(x_o)
        x_o = x_o.view(x_o.shape[0], -1)
        x_o = self.fc(x_o)
        x_o = self.sigmoid(x_o)
        #####
        x_s = self.layer3_s(x)
        x_s = self.layer4_s(x_s)

        x_s = self.avgpool_s(x_s)
        x_s = self.drop(x_s)
        x_s = x_s.view(x_s.shape[0], -1)
        x_s = self.fc1(x_s)
        x_s = self.sigmoid(x_s)
        #####




        return (x_o, x_s)

    def forward_multi(self, x):
        clip_preds = []
        for clip_idx in range(x.shape[1]):  # B, 10, 3, 3, 32, 224, 224
            spatial_crops = []
            for crop_idx in range(x.shape[2]):
                clip = x[:, clip_idx, crop_idx]
                clip = self.forward_single(clip)
                spatial_crops.append(clip)
            spatial_crops = torch.stack(spatial_crops, 1).mean(1)  # (B, 400)
            clip_preds.append(spatial_crops)
        clip_preds = torch.stack(clip_preds, 1).mean(1)  # (B, 400)
        return clip_preds

    def forward(self, batch):  ##x[0] is dark x[1] is light  add permute
        x = batch[0]
        # B, N, C, H, W = x.size()
        # x = x.view(B, C, N, H, W)
        batch = {'frames': x}  ##0 dark 1 light
        # 5D tensor == single clip
        if batch['frames'].dim() == 5:
            pred = self.forward_single(batch['frames'])

        # 7D tensor == 3 crops/10 clips
        elif batch['frames'].dim() == 7:
            pred = self.forward_multi(batch['frames'])

        loss_dict = {}
        if 'label' in batch:
            loss = F.cross_entropy(pred, batch['label'], reduction='none')
            loss_dict = {'loss': loss}

        return pred  # , loss_dict




class ConvBlock(nn.Module):  ###结构提取代码
    def __init__(self, n_in=64, n_out=3):
        super(ConvBlock, self).__init__()

        self.cov_block = nn.Sequential(
            nn.Conv2d(in_channels=n_in, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=n_out, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        # output = []
        # for i in range(x.size(0)):
        #     C,N,H,W = x[i,...].size()
        #     res = self.cov_block(x[i, ...].view(N, C, H, W))
        #     output.append(res)
        # return torch.stack(output)
        res = self.cov_block(x)

        return res

if __name__ == '__main__':
    opt = NetOption()
    xl = torch.randn(4, 21, 3, 244, 244)
    xd = torch.randn(4, 21, 3, 244, 244)
    fullxl = torch.randn(4, 21, 3, 244, 244)
    fullxd = torch.randn(4, 21, 3, 244, 244)
    x = (xl, xd, fullxd, fullxl)
    model = resnet3d()
    a = model(x)


    # extract = ConvBlock(3,3)
    # img = Image.open("/mnt/dataset/CASIA2/3dv_casia2_128slices_newsize/MP-N051_R_CASIA2_LRS_125.jpg").convert ("RGB")
    # trans = transforms.Compose([
    # transforms.Scale(244),
    # transforms.CenterCrop(244),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    # ]
    # )
    #
    # trans1 = transforms.Compose([
    #     transforms.ToPILImage()
    # ]
    # )
    #
    # img = trans(img)
    # c,h,w = img.size()
    # img = extract(img.view(1,c,h,w))
    # img = img.view(c,h,w)
    # img = trans1(img)
    # img.save("/home/yangyifan/save/crop1.jpg")


    # net = i3_res50_nl(400)
    # inp = {'frames': torch.rand(4, 3, 32, 224, 224)}
    # pred, losses = net(inp)
    # if opt.netType =="dual_resnet3d":
    #     model = dual_resnet3d(num_classes=opt.numclass, use_nl=True)
    #     mydict = model.state_dict()
    #     print(opt.pretrain)
    #     a = torch.load(opt.pretrain)
    #     state_dict = torch.load(opt.pretrain)["model"]
    #     # state_dict = model.load_state_dict(a["model"])
    #     print(type(state_dict))
    #     pretrained_dict = {k: v for k, v in state_dict.items() if k not in ["fc.bias", 'fc.weight']}
    #     mydict.update(pretrained_dict)
    #     # a = mydict
    #     model.load_state_dict(mydict)
    #     for p in model.parameters():
    #         p.requires_grad = False
    #     model.fc = nn.Linear(2048, 1)