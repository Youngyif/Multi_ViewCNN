import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from opt.opt import *

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
                self.net2 = retrain.module.net_2
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
        y_l = self.net2 (y_l)
        ysize=y_l.size()
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
        for p in self.parameters():
            p.requires_grad = False
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.drop = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
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

    def forward(self, batch):##x[0] is dark x[1] is light
        x=batch[0]
        B,N,C,H,W = x.size()
        x = x.view(B,C,N,H,W)
        batch = {'frames': x} ##0 dark 1 light
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

        return pred#, loss_dict

def i3_res50_nl(num_classes):
    net = resnet3d(num_classes=num_classes, use_nl=True)
    # state_dict = torch.load('pretrained/i3d_r50_nl_kinetics.pth')
    # net.load_state_dict(state_dict)
    # freeze_bn(net, "net") # Only needed for finetuning. For validation, .eval() works.
    return net
if __name__ == '__main__':
    bot = Bottleneck()
    net = i3_res50_nl(400)
    inp = {'frames': torch.rand(4, 3, 32, 224, 224)}
    pred, losses = net(inp)