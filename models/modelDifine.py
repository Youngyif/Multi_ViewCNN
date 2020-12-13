import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from opt.opt232 import *
# from opt.opt import *
from torch.autograd import Variable
from models.box_filter import BoxFilter
import numpy as np
from torch.nn.init import normal_, constant_

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
class DenseNormReLU(nn.Module):
    def __init__(self, in_feats, out_feats, *args, **kwargs):
        super(DenseNormReLU, self).__init__(*args, **kwargs)
        self.dense = nn.Linear(in_features = in_feats, out_features = out_feats)
        self.bn = nn.BatchNorm1d(out_feats)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.dense(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

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
        # self.conv11 = nn.Conv3d(8192, 2048, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
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
        if opt.contra or opt.contra_focal == True or opt.contra_focal_bilinear == True :
            n=2
        if opt.contra_single==True:
            n=1
        if opt.contra_multiscale:
            n=4
        self.fc = nn.Linear (n * 512 * block.expansion, num_classes)
        # if opt.multifuse==True:
        #     self.fc = nn.Linear(n*256 * block.expansion, num_classes)

        ##### multi scale
        # self.conv11_m = nn.Conv3d(2048, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        # self.conv11_final = nn.Conv3d(4096, 2048, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        ##conv fusion
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
        n = 2
        # self.fc_m = nn.Linear(n * 512 * block.expansion, num_classes)
        n=1
        # self.fc_contra = nn.Sequential(
        #     nn.Linear(n * 512 * block.expansion, 1024),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Linear(1024, 500),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Linear(500, 5))
        # self.fc_contra = nn.Sequential (
        #     nn.Linear (n * 512 * block.expansion, 500),
        #     nn.ReLU (inplace=True),
        #
        #     nn.Linear (500, 5))
        self.drop = nn.Dropout (0.5)
        self.fc_contra_large_scale = nn.Sequential(
            nn.Linear(n * 512 * block.expansion, 500),
            nn.ReLU(inplace=True),
            # #
            nn.Linear(500, 21))

        self.fc_contra_small_scale = nn.Sequential (
            nn.Linear (n * 512 * block.expansion, 500),
            nn.ReLU (inplace=True),
            # #
            nn.Linear (500, 21))

        self.sigmoid = nn.Sigmoid()
        # self.extract = ConvBlock(3,3)
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

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward_single(self, x):
        # print("single, dark")
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
        # print("after avgpool",x.size())
        x = self.drop(x)

        x = x.view(x.shape[0], -1)
        # print("before fc", x.size())
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
        print("x_d_layer1 size", x_d.size())
        x_d = self.layer2(x_d)
        print ("x_d_layer2 size", x_d.size ())
        x_d = self.layer3(x_d)
        print ("x_d_layer3 size", x_d.size ())
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

    def forward_contra_learning_2(self, x):
        x_d,x_l = x
        x_d = self.conv1(x_d)
        x_d = self.bn1(x_d)
        x_d = self.relu(x_d)
        x_d = self.maxpool1(x_d)

        x_d = self.layer1(x_d)
        x_d = self.maxpool2(x_d)
        x_d = self.layer2(x_d)
        x_d = self.layer3(x_d)
        x_d = self.layer4(x_d)

        x_l = self.conv1(x_l)
        x_l = self.bn1(x_l)
        x_l = self.relu(x_l)
        x_l = self.maxpool1(x_l)
        x_l = self.layer1(x_l)
        x_l = self.maxpool2(x_l)
        x_l = self.layer2(x_l)
        x_l = self.layer3(x_l)
        # print("xsize layer3", x_l.size())
        x_l = self.layer4(x_l)
        x = torch.cat((x_l, x_d), dim=1)
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
        # print("contra")
        x_d, x_l = x[2],x[3]
        x_d = self.conv1(x_d)
        x_d = self.bn1(x_d)
        x_d = self.relu(x_d)
        x_d = self.maxpool1(x_d)

        x_d = self.layer1(x_d)
        x_d = self.maxpool2(x_d)
        x_d = self.layer2(x_d)
        # print ("x_d layer2", x_d.size ())
        x_d = self.layer3(x_d)
        # print("x_d layer3", x_d.size())
        x_d = self.layer4(x_d)
        # print("x_d layer4", x_d.size())
        x_d = self.avgpool(x_d)
        # print ("x_d avgpool", x_d.size ())
        ###get similarity vector
        # h_d = x_d
        # h_d = h_d.view(h_d.size(0), -1)
        # h_d = self.fc_contra(h_d)
        # h_d = F.normalize(h_d, dim=1)
        ###
        h_d_3 = self.contra_module (x_d, 3)
        # x_d = self.drop(x_d)

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
        h_l_3 = self.contra_module(x_l, 3)
        # h_l = x_l
        # h_l = h_l.view(h_l.size(0), -1)
        # h_l = self.fc_contra(h_l)
        # h_l = F.normalize(h_l, dim=1)
        ###
        # x_l = self.drop(x_l)

        x = torch.cat((x_l,x_d),dim=1)
        # x = x_d+x_l###sum fuse
        # x = torch.dist(x_d,x_l)
        # x = x_d-x_l
        # x = torch.max(x_l, x_d)
        # x = self.conv11(x)
        x = self.drop(x)
        x = x.view(x.shape[0], -1)

        x = self.fc(x)
        x = self.sigmoid(x)
        return h_d_3, h_l_3, x

    def forward_multiscale_contra(self, x):
        # print("contra")
        x_d, x_l = x[0],x[1]
        x_d = self.conv1(x_d)
        x_d = self.bn1(x_d)
        x_d = self.relu(x_d)
        x_d = self.maxpool1(x_d)

        x_d = self.layer1(x_d)
        x_d = self.maxpool2(x_d)
        x_d = self.layer2(x_d)
        h_d_1 = self.avgpool (x_d)
        # h_d_1 = self.contra_module (h_d_1, 1)
        h_d_1 = self.maxpool1(h_d_1)
        x_d = self.layer3(x_d)
        h_d_2 = self.avgpool(x_d)
        # h_d_2 = self.contra_module (h_d_2, 2)
        x_d = self.layer4(x_d)

        x_d = self.avgpool(x_d)

        # h_d_3 = self.contra_module (x_d, 3)

        x_l = self.conv1(x_l)
        x_l = self.bn1(x_l)
        x_l = self.relu(x_l)
        x_l = self.maxpool1(x_l)
        x_l = self.layer1(x_l)
        x_l = self.maxpool2(x_l)
        x_l = self.layer2(x_l)
        h_l_1 = self.avgpool (x_l)
        # h_l_1 = self.contra_module (h_l_1, 1)
        x_l = self.layer3(x_l)
        h_l_2 = self.avgpool (x_l)
        # h_l_2 = self.contra_module (h_l_2, 2)
        x_l = self.layer4(x_l)
        x_l = self.avgpool(x_l)

        # h_l_3 = self.contra_module(x_l, 3)
        h_l = torch.cat((h_l_1.view(opt.batchSize, -1), h_l_2.view(opt.batchSize, -1), x_l.view(opt.batchSize, -1)))
        h_d = torch.cat((h_d_1.view(opt.batchSize, -1), h_d_2.view(opt.batchSize, -1), x_d.view(opt.batchSize, -1)))
        print(h_d.size())
        print (h_l.size ())
        h_d = self.contra_module(h_d, flag=4)
        h_l = self.contra_module(h_l, flag=4)

        x = torch.cat((x_l,x_d),dim=1)
        # x = self.drop(x)
        x = x.view(x.shape[0], -1)

        x = self.fc(x)
        x = self.sigmoid(x)
        h_d = h_d
        h_l = h_l
        return h_d, h_l, x

    def forward_smallscale(self, x):
        x = self.conv1 (x)
        x = self.bn1 (x)
        x = self.relu (x)
        x = self.maxpool1 (x)

        x = self.layer1 (x)
        x = self.maxpool2 (x)
        x = self.layer2 (x)
        x = self.layer3 (x)
        x = self.layer4 (x)

        x = self.avgpool (x)
        h = x
        h = h.view (h.size (0), -1)
        # h=self.drop(h) ###新增 drop
        h = self.fc_contra_small_scale (h)
        h = F.normalize (h, dim=1)
        # x = self.drop (x)
        #
        # x = x.view (x.shape[0], -1)
        #
        # x = self.fc (x)
        # x = self.sigmoid (x)
        return h, x

    def forward_largescle(self, fullx):
        fullx = self.conv1_m (fullx)
        fullx = self.bn1_m (fullx)
        fullx = self.relu_m (fullx)
        fullx = self.maxpool1_m (fullx)
        fullx = self.layer1_m (fullx)
        fullx = self.maxpool2_m (fullx)
        fullx = self.layer2_m (fullx)
        fullx = self.layer3_m (fullx)
        fullx = self.layer4_m (fullx)
        fullx = self.avgpool (fullx)
        h = fullx
        h = h.view (h.size (0), -1)
        h=self.drop(h)###新增 drop
        h = self.fc_contra_large_scale (h)
        h = F.normalize (h, dim=1)

        # fullx = self.drop (fullx)
        # fullx = fullx.view (fullx.shape[0], -1)
        # fullx = self.fc (fullx)
        # fullx = self.sigmoid (fullx)
        return h, fullx

    def forward_single_mscale_single(self, input):
        x_d, x_l, fullx_d, fullx_l = input
        h_full_d, x_full_d = self.forward_largescle (fullx_d)
        h_full_l, x_full_l = self.forward_largescle (fullx_l)
        h_d, x_d = self.forward_smallscale (x_d)
        h_l, x_l = self.forward_smallscale (x_l)
        full_x = torch.cat((x_full_d, x_full_l),dim=1)
        # full_x = self.drop (full_x)
        # full_x = full_x.view (full_x.shape[0], -1)
        # print(full_x.size())
        # full_x = self.fc_m (full_x)
        # full_x = self.sigmoid (full_x)
        x = torch.cat ((x_d, x_l), dim=1)
        # print(x.size(), full_x.size())
        x = torch.cat((full_x,x),dim=1)
        # x = self.conv11(x)
        x = self.drop (x)

        x = x.view (x.shape[0], -1)

        x = self.fc (x)
        x = self.sigmoid (x)
        # x = (x+full_x)/2

        return {"h_full_d": h_full_d, "h_full_l": h_full_l, "h_d": h_d, "h_l": h_l,"x": x}

    def contra_module(self, x, flag):
        h = x
        h = h.view (h.size (0), -1)
        if flag==1:
            h = self.fc_contra_scale1 (h)
        elif flag==2:
            h = self.fc_contra_scale2 (h)
        elif flag==3:
            h = self.fc_contra (h)
        elif flag==4:
            h = self.fc_contra_multi (h)
        h = F.normalize (h, dim=1)
        return h


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

    def forward(self, batch):##x[0] is dark    x[1] is light Pair = (dark_input_var, light_input_var, fulldark_var, fulllight_var)
        x_d = batch[0]
        x_l = batch[1]
        fullx_d = batch[2]
        fullx_l = batch[3]
        # x = (x_d, x_l)
        ms_x = (x_d, x_l, fullx_d, fullx_l)
        batch = {'frames1': ms_x} ##0 dark 1 light
        # 5D tensor == single clip
        # if opt.multiway_contra == True:
        #     pred = self.forward_multiway_contra (batch['frames'])
        # if opt.contra_focal_bilinear ==True:
        #     pred = self.forward_single_contra_bilinear(batch['frames'])
        # if opt.contra_single == True:
        #     # pred = self.forward_single_contra(batch['frames'])
        #     # print("batchsize",batch['frames'][0].size())
        #     pred = self.forward_single(batch['frames'][0]) ##0 dark 1 light
        # if opt.contra_learning == True:
        #     pred = self.forward_single_contra_learning(batch['frames'])
        # if opt.contra_learning_2 == True:
        #     print("contra_learning")
        #     pred = self.forward_contra_learning_2(batch['frames'])
        if opt.mscale == True:###what is that
            # print("multiscale cat")
            pred = self.forward_single_mscale_single(batch['frames1'])
            # pred = self.forward_single_mscale(batch['frames1'])
        # if opt.cat == True:
        #     # if batch['frames'].dim() == 5:
        #     # print("catmodel")
        #     pred = self.forward_single_cat(batch['frames'])
        # if  opt.contra_multiscale and opt.contra_focal:
        #     # print("multiscal_contra")
        #     pred = self.forward_single_mscale_single(batch['frames1'])
        # if  not opt.contra_multiscale and opt.contra_focal:
        #     # print("multiscal_contra")
        #     pred = self.forward_single_contra(batch['frames1'])

        return pred

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

class C3D (nn.Module):
        """
        The C3D network as described in [1].
        """

        def __init__(self):
            super (C3D, self).__init__ ()

            self.conv1 = nn.Conv3d (3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool1 = nn.MaxPool3d (kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.conv2 = nn.Conv3d (64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool2 = nn.MaxPool3d (kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv3a = nn.Conv3d (128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv3b = nn.Conv3d (256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool3 = nn.MaxPool3d (kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv4a = nn.Conv3d (256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv4b = nn.Conv3d (512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool4 = nn.MaxPool3d (kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv5a = nn.Conv3d (512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv5b = nn.Conv3d (512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool5 = nn.MaxPool3d (kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
            self.convfuse = nn.Conv3d (1024, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                                       bias=False)
            self.fc6 = nn.Linear (8192, 4096)
            self.fc7 = nn.Linear (4096, 4096)
            self.fc8 = nn.Linear (4096, 1)

            self.dropout = nn.Dropout (p=0.5)

            self.relu = nn.ReLU ()
            # self.softmax = nn.Softmax()
            self.sigmoid = nn.Sigmoid ()

            self.fc_contra = nn.Sequential (
                nn.Linear (8192, 500),
                nn.ReLU (inplace=True),

                nn.Linear (500, 500),
                nn.ReLU (inplace=True),

                nn.Linear (500, 5))

        def forward_single(self, x):  # x # ch, fr, h, w
            h = self.relu (self.conv1 (x))
            h = self.pool1 (h)

            h = self.relu (self.conv2 (h))
            h = self.pool2 (h)

            h = self.relu (self.conv3a (h))
            h = self.relu (self.conv3b (h))
            h = self.pool3 (h)

            h = self.relu (self.conv4a (h))
            h = self.relu (self.conv4b (h))
            h = self.pool4 (h)

            h = self.relu (self.conv5a (h))
            h = self.relu (self.conv5b (h))
            h = self.pool5 (h)
            h = h.view (-1, 8192)
            h = self.relu (self.fc6 (h))
            h = self.dropout (h)
            h = self.relu (self.fc7 (h))
            h = self.dropout (h)
            logits = self.fc8 (h)
            probs = self.sigmoid (logits)
            return probs

        def feature(self, x):
            h = self.relu (self.conv1 (x))
            h = self.pool1 (h)

            h = self.relu (self.conv2 (h))
            h = self.pool2 (h)

            h = self.relu (self.conv3a (h))
            h = self.relu (self.conv3b (h))
            h = self.pool3 (h)

            h = self.relu (self.conv4a (h))
            h = self.relu (self.conv4b (h))
            h = self.pool4 (h)

            h = self.relu (self.conv5a (h))
            h = self.relu (self.conv5b (h))
            # print(h.size())
            h = self.pool5 (h)
            # print(h.size())
            return h

        def forward_contra(self, x):  # x # ch, fr, h, w
            x_d, x_l = x[0], x[1]
            h_d = self.feature (x_d)
            h_l = self.feature (x_l)
            ####
            x_l = h_l.view (h_l.shape[0], -1)
            x_d = h_d.view (h_d.shape[0], -1)
            x_l = self.fc_contra (x_l)
            x_l = F.normalize (x_l, dim=1)
            x_d = self.fc_contra (x_d)
            x_d = F.normalize (x_d, dim=1)
            ###
            h = torch.cat ((h_d, h_l), dim=1)
            h = self.convfuse (h)
            h = h.view (-1, 8192)
            h = self.relu (self.fc6 (h))
            h = self.dropout (h)
            h = self.relu (self.fc7 (h))
            h = self.dropout (h)
            logits = self.fc8 (h)
            probs = self.sigmoid (logits)
            return x_d, x_l, probs

        def forward(self, x):
            if opt.contra_focal == True:
                print ("c3d focal ")
                return self.forward_contra (x)
            elif opt.contra_single == True:
                return self.forward_single (x)


class MaxPool3dSamePadding (nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max (self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max (self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size ()
        # print t,h,w
        out_t = np.ceil (float (t) / float (self.stride[0]))
        out_h = np.ceil (float (h) / float (self.stride[1]))
        out_w = np.ceil (float (w) / float (self.stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad (0, t)
        pad_h = self.compute_pad (1, h)
        pad_w = self.compute_pad (2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print (x.size())
        # print pad
        x = F.pad (x, pad)
        return super (MaxPool3dSamePadding, self).forward (x)


class Unit3D (nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):

        """Initializes Unit3D module."""
        super (Unit3D, self).__init__ ()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        self.conv3d = nn.Conv3d (in_channels=in_channels,
                                 out_channels=self._output_channels,
                                 kernel_size=self._kernel_shape,
                                 stride=self._stride,
                                 padding=0,
                                 # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                 bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d (self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max (self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max (self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size ()
        # print t,h,w
        out_t = np.ceil (float (t) / float (self._stride[0]))
        out_h = np.ceil (float (h) / float (self._stride[1]))
        out_w = np.ceil (float (w) / float (self._stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad (0, t)
        pad_h = self.compute_pad (1, h)
        pad_w = self.compute_pad (2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad (x, pad)
        # print x.size()

        x = self.conv3d (x)
        if self._use_batch_norm:
            x = self.bn (x)
        if self._activation_fn is not None:
            x = self._activation_fn (x)
        return x


class InceptionModule (nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super (InceptionModule, self).__init__ ()

        self.b0 = Unit3D (in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D (in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                           name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D (in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                           name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D (in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                           name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D (in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                           name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding (kernel_size=[3, 3, 3],
                                         stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D (in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                           name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0 (x)
        b1 = self.b1b (self.b1a (x))
        b2 = self.b2b (self.b2a (x))
        b3 = self.b3b (self.b3a (x))
        return torch.cat ([b0, b1, b2, b3], dim=1)


"""
 def load_state_dict(self, path):
        target_weights = torch.load(path)
        own_state = self.state_dict()
        for name, param in target_weights.items():
            if '.bn.' not in name:
                try:
                    s3d_name = i3d_to_s3d[name]
                    if param.size()[-1] != 1:
                        param = torch.mean(param, 2, keepdim=True)
                    else:
                        param = param.data
                    own_state[s3d_name].copy_(param)
                    print('Copied param: {}'.format(s3d_name))
                except KeyError:
                    print('No data for param: {}'.format(name))
"""


class S3D (nn.Module):
    def __init__(self, num_class):
        print ("init S3D")
        super (S3D, self).__init__ ()
        self.sigmoid = nn.Sigmoid ()
        self.base = nn.Sequential (
            SepConv3d (3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d (kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            BasicConv3d (64, 64, kernel_size=1, stride=1),
            SepConv3d (64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d (kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            Mixed_3b (),
            Mixed_3c (),
            nn.MaxPool3d (kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            Mixed_4b (),
            Mixed_4c (),
            Mixed_4d (),
            Mixed_4e (),
            Mixed_4f (),
            nn.MaxPool3d (kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)),
            Mixed_5b (),
            Mixed_5c (),
        )
        self.base_m = nn.Sequential (
            SepConv3d (3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d (kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            BasicConv3d (64, 64, kernel_size=1, stride=1),
            SepConv3d (64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d (kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            Mixed_3b (),
            Mixed_3c (),
            nn.MaxPool3d (kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            Mixed_4b (),
            Mixed_4c (),
            Mixed_4d (),
            Mixed_4e (),
            Mixed_4f (),
            nn.MaxPool3d (kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)),
            Mixed_5b (),
            Mixed_5c (),
        )
        n = 2048
        self.fc_contra_m = nn.Sequential (
            nn.Linear (n, 500),
            nn.ReLU (inplace=True),

            nn.Linear (500, 21))
        self.fc_contra= nn.Sequential (
            nn.Linear (n, 500),
            nn.ReLU (inplace=True),

            nn.Linear (500, 21))
        self.fc = nn.Sequential (nn.Conv3d (1024, num_class, kernel_size=1, stride=1, bias=True), )
        self.fc_2 = nn.Sequential (nn.Conv3d (2048, num_class, kernel_size=1, stride=1, bias=True), )
    # def forward_single(self, x):
    #     y = self.base (x)
    #     print("xsize",y.size())
    #     y = F.avg_pool3d (y, (2, y.size (3), y.size (4)), stride=1)
    #     print ("xsize", y.size ())
    #     y = self.fc (y)
    #     print("899", y.size())
    #     y = y.view (y.size (0), y.size (1), y.size (2))
    #     print("901", y.size(), y)
    #     logits = torch.mean (y, 2)
    #
    #     logits = self.sigmoid (logits)
    #     print("903", logits.size())
    #     return logits
    # def forward_single_scale(self, x): ##forward contra org branch
    #     x_d,x_l = x[0],x[1]
    #     y_d = self.base (x_d)
    #     y_d = F.avg_pool3d (y_d, (2, y_d.size (3), y_d.size (4)), stride=1)
    #     h_d = self.fc_contra(y_d.view(y_d.size(0),-1))
    #     y_l = self.base (x_l)
    #     y_l = F.avg_pool3d (y_l, (2, y_l.size (3), y_l.size (4)), stride=1)
    #     h_l = self.fc_contra (y_l.view (y_l.size (0), -1))
    #
    #     y = torch.cat((y_d, y_l),dim=1)
    #
    #     y = self.fc_2 (y)
    #
    #     y = y.view (y.size (0), y.size (1), y.size (2))
    #
    #     logits = torch.mean (y, 2)
    #
    #     logits = self.sigmoid (logits)
    #
    #     return h_d, h_l, logits


    def forward_single_light_small(self, x):
        y = self.base (x)
        y = F.avg_pool3d (y, (2, y.size (3), y.size (4)), stride=1)
        h=y
        h = self.fc_contra_m (h.view (h.size (0), -1))
        return h,y

    def forward_single_light_large(self,x):
        y = self.base_m (x)
        y = F.avg_pool3d (y, (2, y.size (3), y.size (4)), stride=1)
        h=y
        h = self.fc_contra_m (h.view (h.size (0), -1))
        return h,y
    def forward_msscale(self,x):
        x_d, x_l, full_x_d, full_x_l = x[0], x[1], x[2], x[3]
        h_d, x_d = self.forward_single_light_small (x_d)
        h_l, x_l = self.forward_single_light_small (x_l)
        h_full_l, full_x_l = self.forward_single_light_large (full_x_l)
        h_full_d, full_x_d = self.forward_single_light_large (full_x_d)
        full_x = torch.cat ((full_x_d, full_x_l), dim=1)
        x = torch.cat ((x_d, x_l), dim=1)
        y = torch.cat ((full_x, x), dim=1)
        y = self.fc_2 (y)

        y = y.view (y.size (0), y.size (1), y.size (2))

        logits = torch.mean (y, 2)

        logits = self.sigmoid (logits)

        return {"h_full_d": h_full_d, "h_full_l": h_full_l, "h_d": h_d, "h_l": h_l,"x": logits}

    def forward(self,x):
        print("msscale")
        return self.forward_msscale(x)

class BasicConv3d (nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super (BasicConv3d, self).__init__ ()
        self.conv = nn.Conv3d (in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=False)
        self.bn = nn.BatchNorm3d (out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU ()

    def forward(self, x):
        x = self.conv (x)
        x = self.bn (x)
        x = self.relu (x)
        return x


class SepConv3d (nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super (SepConv3d, self).__init__ ()
        self.conv_s = nn.Conv3d (in_planes, out_planes, kernel_size=(1, kernel_size, kernel_size),
                                 stride=(1, stride, stride), padding=(0, padding, padding), bias=False)
        self.bn_s = nn.BatchNorm3d (out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_s = nn.ReLU ()

        self.conv_t = nn.Conv3d (out_planes, out_planes, kernel_size=(kernel_size, 1, 1), stride=(stride, 1, 1),
                                 padding=(padding, 0, 0), bias=False)
        self.bn_t = nn.BatchNorm3d (out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_t = nn.ReLU ()

    def forward(self, x):
        x = self.conv_s (x)
        x = self.bn_s (x)
        x = self.relu_s (x)

        x = self.conv_t (x)
        x = self.bn_t (x)
        x = self.relu_t (x)
        return x


class Mixed_3b (nn.Module):
    def __init__(self):
        super (Mixed_3b, self).__init__ ()

        self.branch0 = nn.Sequential (
            BasicConv3d (192, 64, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential (
            BasicConv3d (192, 96, kernel_size=1, stride=1),
            SepConv3d (96, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential (
            BasicConv3d (192, 16, kernel_size=1, stride=1),
            SepConv3d (16, 32, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential (
            nn.MaxPool3d (kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d (192, 32, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0 (x)
        x1 = self.branch1 (x)
        x2 = self.branch2 (x)
        x3 = self.branch3 (x)
        out = torch.cat ((x0, x1, x2, x3), 1)

        return out


class Mixed_3c (nn.Module):
    def __init__(self):
        super (Mixed_3c, self).__init__ ()
        self.branch0 = nn.Sequential (
            BasicConv3d (256, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential (
            BasicConv3d (256, 128, kernel_size=1, stride=1),
            SepConv3d (128, 192, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential (
            BasicConv3d (256, 32, kernel_size=1, stride=1),
            SepConv3d (32, 96, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential (
            nn.MaxPool3d (kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d (256, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0 (x)
        x1 = self.branch1 (x)
        x2 = self.branch2 (x)
        x3 = self.branch3 (x)
        out = torch.cat ((x0, x1, x2, x3), 1)
        return out


class Mixed_4b (nn.Module):
    def __init__(self):
        super (Mixed_4b, self).__init__ ()

        self.branch0 = nn.Sequential (
            BasicConv3d (480, 192, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential (
            BasicConv3d (480, 96, kernel_size=1, stride=1),
            SepConv3d (96, 208, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential (
            BasicConv3d (480, 16, kernel_size=1, stride=1),
            SepConv3d (16, 48, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential (
            nn.MaxPool3d (kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d (480, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0 (x)
        x1 = self.branch1 (x)
        x2 = self.branch2 (x)
        x3 = self.branch3 (x)
        out = torch.cat ((x0, x1, x2, x3), 1)
        return out


class Mixed_4c (nn.Module):
    def __init__(self):
        super (Mixed_4c, self).__init__ ()

        self.branch0 = nn.Sequential (
            BasicConv3d (512, 160, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential (
            BasicConv3d (512, 112, kernel_size=1, stride=1),
            SepConv3d (112, 224, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential (
            BasicConv3d (512, 24, kernel_size=1, stride=1),
            SepConv3d (24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential (
            nn.MaxPool3d (kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d (512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0 (x)
        x1 = self.branch1 (x)
        x2 = self.branch2 (x)
        x3 = self.branch3 (x)
        out = torch.cat ((x0, x1, x2, x3), 1)
        return out


class Mixed_4d (nn.Module):
    def __init__(self):
        super (Mixed_4d, self).__init__ ()

        self.branch0 = nn.Sequential (
            BasicConv3d (512, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential (
            BasicConv3d (512, 128, kernel_size=1, stride=1),
            SepConv3d (128, 256, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential (
            BasicConv3d (512, 24, kernel_size=1, stride=1),
            SepConv3d (24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential (
            nn.MaxPool3d (kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d (512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0 (x)
        x1 = self.branch1 (x)
        x2 = self.branch2 (x)
        x3 = self.branch3 (x)
        out = torch.cat ((x0, x1, x2, x3), 1)
        return out


class Mixed_4e (nn.Module):
    def __init__(self):
        super (Mixed_4e, self).__init__ ()

        self.branch0 = nn.Sequential (
            BasicConv3d (512, 112, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential (
            BasicConv3d (512, 144, kernel_size=1, stride=1),
            SepConv3d (144, 288, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential (
            BasicConv3d (512, 32, kernel_size=1, stride=1),
            SepConv3d (32, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential (
            nn.MaxPool3d (kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d (512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0 (x)
        x1 = self.branch1 (x)
        x2 = self.branch2 (x)
        x3 = self.branch3 (x)
        out = torch.cat ((x0, x1, x2, x3), 1)
        return out


class Mixed_4f (nn.Module):
    def __init__(self):
        super (Mixed_4f, self).__init__ ()

        self.branch0 = nn.Sequential (
            BasicConv3d (528, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential (
            BasicConv3d (528, 160, kernel_size=1, stride=1),
            SepConv3d (160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential (
            BasicConv3d (528, 32, kernel_size=1, stride=1),
            SepConv3d (32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential (
            nn.MaxPool3d (kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d (528, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0 (x)
        x1 = self.branch1 (x)
        x2 = self.branch2 (x)
        x3 = self.branch3 (x)
        out = torch.cat ((x0, x1, x2, x3), 1)
        return out


class Mixed_5b (nn.Module):
    def __init__(self):
        super (Mixed_5b, self).__init__ ()

        self.branch0 = nn.Sequential (
            BasicConv3d (832, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential (
            BasicConv3d (832, 160, kernel_size=1, stride=1),
            SepConv3d (160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential (
            BasicConv3d (832, 32, kernel_size=1, stride=1),
            SepConv3d (32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential (
            nn.MaxPool3d (kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d (832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0 (x)
        x1 = self.branch1 (x)
        x2 = self.branch2 (x)
        x3 = self.branch3 (x)
        out = torch.cat ((x0, x1, x2, x3), 1)
        return out


class Mixed_5c (nn.Module):
    def __init__(self):
        super (Mixed_5c, self).__init__ ()

        self.branch0 = nn.Sequential (
            BasicConv3d (832, 384, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential (
            BasicConv3d (832, 192, kernel_size=1, stride=1),
            SepConv3d (192, 384, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential (
            BasicConv3d (832, 48, kernel_size=1, stride=1),
            SepConv3d (48, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential (
            nn.MaxPool3d (kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d (832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0 (x)
        x1 = self.branch1 (x)
        x2 = self.branch2 (x)
        x3 = self.branch3 (x)
        out = torch.cat ((x0, x1, x2, x3), 1)
        return out


class InceptionI3d (nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )
    VALID_ENDPOINTS_L = (
        'Conv3d_1a_7x7_l',
        'MaxPool3d_2a_3x3_l',
        'Conv3d_2b_1x1_l',
        'Conv3d_2c_3x3_l',
        'MaxPool3d_3a_3x3_l',
        'Mixed_3b_l',
        'Mixed_3c_l',
        'MaxPool3d_4a_3x3_l',
        'Mixed_4b_l',
        'Mixed_4c_l',
        'Mixed_4d_l',
        'Mixed_4e_l',
        'Mixed_4f_l',
        'MaxPool3d_5a_2x2_l',
        'Mixed_5b_l',
        'Mixed_5c_l',
        'Logits_l',
        'Predictions_l',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError ('Unknown final endpoint %s' % final_endpoint)

        super (InceptionI3d, self).__init__ ()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None
        #######
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError ('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D (in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                             stride=(2, 2, 2), padding=(3, 3, 3), name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding (kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                           padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D (in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                             name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D (in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                             name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding (kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                           padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule (192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule (256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding (kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                           padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule (128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule (192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64],
                                                      name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule (160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64],
                                                      name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule (128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64],
                                                      name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule (112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128],
                                                      name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding (kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                           padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule (256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],
                                                      name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule (256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],
                                                      name + end_point)
        if self._final_endpoint == end_point: return

#########come on##########
        self.end_points_l={}
        end_point = 'Conv3d_1a_7x7_l'
        self.end_points_l[end_point] = Unit3D (in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                             stride=(2, 2, 2), padding=(3, 3, 3), name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_2a_3x3_l'
        self.end_points_l[end_point] = MaxPool3dSamePadding (kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                           padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2b_1x1_l'
        self.end_points_l[end_point] = Unit3D (in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                             name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2c_3x3_l'
        self.end_points_l[end_point] = Unit3D (in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                             name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3_l'
        self.end_points_l[end_point] = MaxPool3dSamePadding (kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                           padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3b_l'
        self.end_points_l[end_point] = InceptionModule (192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c_l'
        self.end_points_l[end_point] = InceptionModule (256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3_l'
        self.end_points_l[end_point] = MaxPool3dSamePadding (kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                           padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b_l'
        self.end_points_l[end_point] = InceptionModule (128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c_l'
        self.end_points_l[end_point] = InceptionModule (192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64],
                                                      name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d_l'
        self.end_points_l[end_point] = InceptionModule (160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64],
                                                      name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e_l'
        self.end_points_l[end_point] = InceptionModule (128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64],
                                                      name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f_l'
        self.end_points_l[end_point] = InceptionModule (112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128],
                                                      name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2_l'
        self.end_points_l[end_point] = MaxPool3dSamePadding (kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                           padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b_l'
        self.end_points_l[end_point] = InceptionModule (256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],
                                                      name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c_l'
        self.end_points_l[end_point] = InceptionModule (256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],
                                                      name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d (kernel_size=[2, 7, 7],
                                      stride=(1, 1, 1))
        self.dropout = nn.Dropout (dropout_keep_prob)
        self.logits = Unit3D (in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes, #4*(384 + 384 + 128 + 128)
                              kernel_shape=[1, 1, 1],
                              padding=0,
                              activation_fn=None,
                              use_batch_norm=False,
                              use_bias=True,
                              name='logits')
        self.sigmoid = nn.Sigmoid ()
        n = 2048
        self.conv11 = nn.Conv3d (4096, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.fc_contra_small = nn.Sequential (
            nn.Linear (n , 500),
            nn.ReLU (inplace=True),

            nn.Linear (500, 21))
        self.fc_contra_large = nn.Sequential (
            nn.Linear (n, 500),
            nn.ReLU (inplace=True),

            nn.Linear (500, 21))
        # self.fc_contra = nn.Sequential (
        #     nn.Linear (n, 500),
        #     nn.ReLU (inplace=True),
        #
        #     nn.Linear (500, 500),
        #     nn.ReLU (inplace=True),
        #
        #     nn.Linear (500, 5))

        self.build ()

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D (in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                              kernel_shape=[1, 1, 1],
                              padding=0,
                              activation_fn=None,
                              use_batch_norm=False,
                              use_bias=True,
                              name='logits')

    def build(self):
        for k in self.end_points.keys ():
            self.add_module (k, self.end_points[k])
        for k in self.end_points_l.keys ():
            self.add_module (k, self.end_points_l[k])

    def forward_single(self, x):
        print("i3d",x.size())
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point] (x)  # use _modules to work with dataparallel
        ##add contrastive model

        x = self.logits (self.dropout (self.avg_pool (x)))
        if self._spatial_squeeze:
            logits = x.squeeze (3).squeeze (3)
        # logits is batch X time X classes, which is what we want to work with
        t = logits.size (2)
        per_frame_logits = F.upsample (logits, t, mode='linear')
        # print(per_frame_logits.size())
        average_logits = torch.mean (per_frame_logits, 2)
        average_logits = self.sigmoid (average_logits)
        # print("average_size",average_logits.size())
        return average_logits



    def forward_contra(self, x):
        x_d, x_l = x[0], x[1]
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x_l = self._modules[end_point] (x_l)
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x_d = self._modules[end_point] (x_d)  # use _modules to work with dataparallel
        ##

        ##add contrastive model
        # print (x_d.size ())
        x_d = self.avg_pool (x_d)
        x_l = self.avg_pool (x_l)
        h_d, h_l = x_d, x_l
        x_d,x_l = self.dropout (x_d), self.dropout(x_l)
        # print(h_d.size())
        h_d = h_d.view (h_d.size (0), -1)
        h_d = self.fc_contra (h_d)
        h_d = F.normalize (h_d, dim=1)
        h_l = h_l.view (h_l.size (0), -1)
        h_l = self.fc_contra (h_l)
        h_l = F.normalize (h_l, dim=1)
        # if opt.mixup:
        #     lam = np.random.beta(opt.alpha, opt.alpha)
        #     x = lam*x_l+(1.-lam)*x_l
        # elif opt.singleinput:
        #     x = x_l##仅仅使用x_l 的信息

        x = torch.cat ((x_l, x_d), dim=1)
        x = self.conv11 (x)
        x = self.logits (x)

        if self._spatial_squeeze:
            logits = x.squeeze (3).squeeze (3)
        # logits is batch X time X classes, which is what we want to work with
        t = logits.size (2)
        per_frame_logits = F.upsample (logits, t, mode='linear')
        average_logits = torch.mean (per_frame_logits, 2)
        average_logits = self.sigmoid (average_logits)

        h_d = 10*h_d
        h_l = 10*h_l
        return h_d, h_l, average_logits



    def forward_single_light_small(self,x):
        # print ("i3d", x.size ())
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point] (x)  # use _modules to work with dataparallel
        ##add contrastive model
        x = self.avg_pool (x)
        h=x
        h = h.view (h.size (0), -1)
        h = self.fc_contra_small (h)
        h = F.normalize (h, dim=1)
        # x = self.logits (self.dropout (x))
        # if self._spatial_squeeze:
        #     logits = x.squeeze (3).squeeze (3)
        # # logits is batch X time X classes, which is what we want to work with
        # t = logits.size (2)
        # per_frame_logits = F.upsample (logits, t, mode='linear')
        # average_logits = torch.mean (per_frame_logits, 2)
        # average_logits = self.sigmoid (average_logits)
        return h, x

    def forward_single_light_large(self, x):
        # print ("i3d", x.size ())
        # print ("large scale")
        for end_point in self.VALID_ENDPOINTS_L:
            if end_point in self.end_points_l:
                # print(end_point)
                x = self._modules[end_point] (x)  # use _modules to work with dataparallel
        ##add contrastive model
        x = self.avg_pool (x)
        h = x
        h = h.view (h.size (0), -1)
        h = self.fc_contra_large (h)
        h = F.normalize (h, dim=1)
        # x = self.logits (self.dropout (x))
        # if self._spatial_squeeze:
        #     logits = x.squeeze (3).squeeze (3)
        # # logits is batch X time X classes, which is what we want to work with
        # t = logits.size (2)
        # per_frame_logits = F.upsample (logits, t, mode='linear')
        # average_logits = torch.mean (per_frame_logits, 2)
        # average_logits = self.sigmoid (average_logits)
        return h, x

    def forward(self, x):
        if opt.contra_focal == True and not opt.contra_multiscale :
            return self.forward_contra (x)
        elif opt.contra_focal == True and opt.contra_multiscale:
            x_d, x_l, full_x_d, full_x_l = x[0], x[1], x[2], x[3]
            h_d, x_d = self.forward_single_light_small (x_d)
            h_l, x_l = self.forward_single_light_small (x_l)
            h_full_l, full_x_l = self.forward_single_light_large (full_x_l)
            h_full_d, full_x_d = self.forward_single_light_large (full_x_d)
            full_x = torch.cat ((full_x_d, full_x_l), dim=1)
            x = torch.cat ((x_d, x_l), dim=1)
            x = torch.cat ((full_x, x), dim=1)
            x = self.conv11(x)
            x = self.logits (self.dropout (x))
            if self._spatial_squeeze:
                logits = x.squeeze (3).squeeze (3)
            # logits is batch X time X classes, which is what we want to work with
            t = logits.size (2)
            per_frame_logits = F.upsample (logits, t, mode='linear')
            # print(per_frame_logits.size())
            average_logits = torch.mean (per_frame_logits, 2)
            average_logits = self.sigmoid (average_logits)
            # print("average_size",average_logits.size())
            return  {"h_full_d": h_full_d, "h_full_l": h_full_l, "h_d": h_d, "h_l": h_l,"x": average_logits}
        elif opt.contra_single == True:

            return self.forward_single (x)

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point] (x)
        return self.avg_pool (x)


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



class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=False,
                 dropout=0.8, img_feature_dim=256,
                 crop_num=1, partial_bn=True, print_spec=True, pretrain='imagenet',
                 is_shift=False, shift_div=8, shift_place='blockres', fc_lr5=False,
                 temporal_pool=False, non_local=False):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        self.pretrain = pretrain

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool
        self.non_local = non_local

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        if print_spec:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout, self.img_feature_dim)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()
            self.sigmoid = nn.Sigmoid()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))

        if 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True if self.pretrain == 'imagenet' else False)
            if self.is_shift:
                print('Adding temporal shift...')
                from ops.temporal_shift import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            if self.non_local:
                print('Adding non-local module...')
                from ops.non_local import make_non_local
                make_non_local(self.base_model, self.num_segments)

            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'mobilenetv2':
            from archs.mobilenet_v2 import mobilenet_v2, InvertedResidual
            self.base_model = mobilenet_v2(True if self.pretrain == 'imagenet' else False)

            self.base_model.last_layer_name = 'classifier'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
            if self.is_shift:
                from ops.temporal_shift import TemporalShift
                for m in self.base_model.modules():
                    if isinstance(m, InvertedResidual) and len(m.conv) == 8 and m.use_res_connect:
                        if self.print_spec:
                            print('Adding temporal shift... {}'.format(m.use_res_connect))
                        m.conv[0] = TemporalShift(m.conv[0], n_segment=self.num_segments, n_div=self.shift_div)
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'BNInception':
            from archs.bn_inception import bninception
            self.base_model = bninception(pretrained=self.pretrain)
            self.input_size = self.base_model.input_size
            self.input_mean = self.base_model.mean
            self.input_std = self.base_model.std
            self.base_model.last_layer_name = 'fc'
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
            if self.is_shift:
                print('Adding temporal shift...')
                self.base_model.build_temporal_ops(
                    self.num_segments, is_temporal_shift=self.shift_place, shift_div=self.shift_div)
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]

    def forward(self, input, no_reshape=False):
        if not no_reshape:
            sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

            if self.modality == 'RGBDiff':
                sample_len = 3 * self.new_length
                input = self._get_diff(input)

            base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
        else:
            base_out = self.base_model(input)
        print ("baseoutsize", base_out.size())
        base_out1 = base_out.view (4,-1)
        print ("baseout1", base_out1.size ())
        if self.dropout > 0:
            base_out = self.new_fc(base_out)
        print ("baseoutsize1", base_out.size ())
        if not self.before_softmax:
            # print("True  sigmoid")
            # base_out = self.softmax(base_out)
            base_out = self.sigmoid(base_out)
        print ("baseoutsize2", base_out.size ())
        if self.reshape:
            if self.is_shift and self.temporal_pool:
                base_out = base_out.view((-1, self.num_segments // 2) + base_out.size()[1:])
            else:
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            print ("baseoutsize3", base_out.size ())
            output = self.consensus(base_out)
            print ("baseoutsize4", output.size ())
            return output.squeeze(1)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

        if self.base_model_name == 'BNInception':
            import torch.utils.model_zoo as model_zoo
            sd = model_zoo.load_url('https://www.dropbox.com/s/35ftw2t4mxxgjae/BNInceptionFlow-ef652051.pth.tar?dl=1')
            base_model.load_state_dict(sd)
            print('=> Loading pretrained Flow weight done...')
        else:
            print('#' * 30, 'Warning! No Flow pretrained model is found')
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if self.modality == 'RGB':
            if flip:
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip(is_flow=False)])
            else:
                print('#' * 20, 'NO FLIP!!!')
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])


if __name__ == '__main__':
    print("load demo")
    base = nn.Sequential (
        SepConv3d (3, 64, kernel_size=7, stride=2, padding=3),
        nn.MaxPool3d (kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        BasicConv3d (64, 64, kernel_size=1, stride=1),
        SepConv3d (64, 192, kernel_size=3, stride=1, padding=1),
        nn.MaxPool3d (kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        Mixed_3b (),
        Mixed_3c (),
        nn.MaxPool3d (kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
        Mixed_4b (),
        Mixed_4c (),
        Mixed_4d (),
        Mixed_4e (),
        Mixed_4f (),
        nn.MaxPool3d (kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)),
        Mixed_5b (),
        Mixed_5c (),
    )
    for par in base[-1].parameters():
        print(par)

    print("################################")
    base_m = nn.Sequential (
        SepConv3d (3, 64, kernel_size=7, stride=2, padding=3),
        nn.MaxPool3d (kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        BasicConv3d (64, 64, kernel_size=1, stride=1),
        SepConv3d (64, 192, kernel_size=3, stride=1, padding=1),
        nn.MaxPool3d (kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        Mixed_3b (),
        Mixed_3c (),
        nn.MaxPool3d (kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
        Mixed_4b (),
        Mixed_4c (),
        Mixed_4d (),
        Mixed_4e (),
        Mixed_4f (),
        nn.MaxPool3d (kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)),
        Mixed_5b (),
        Mixed_5c (),
    )
    for par in base_m[-1].parameters():
        print(par)

    # opt = NetOption()
    # xl = torch.randn(4, 3, 21, 244, 244)
    # xd = torch.randn(4, 3, 21, 244, 244)
    # fullxl = torch.randn(4, 3, 21, 244, 244)
    # fullxd = torch.randn(4, 3, 21, 244, 244)
    # x = (xl, xd, fullxd, fullxl)
    # # x=(xl,xd)
    # model = resnet3d()
    # pred = model(x)
    # print(a.size(),  h[0].size(), h[1].size())
    # input_shape = (2,3, 21, 244, 244)
    # # model.replace_logits(1)
    #
    #
    # summary (model, input_shape)

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

    # num_class = 1
    #
    # net  = TSN(num_class, 21, "RGB",
    #           base_model='resnet101',
    #           consensus_type="avg",
    #           img_feature_dim=244,
    #           pretrain='imagenet',
    #           is_shift=False, shift_div=21, shift_place='blockres',
    #           non_local=False,
    #           )
    #
    # a =  torch.randn(4, 3, 21, 244, 244)
    # b = net(a)
    # print(b)
    # model = S3D (opt.numclass)
    # a = torch.randn (4, 3, 21, 244, 244)
    # b = model((a,a))
    # print(b[0].size(), b[1].size(), b[2].size())