import torch
import torch.nn.functional as F
import torch.nn as nn
import math


def cosine_distance(prototype1, prototype2):
    similarity = prototype1.mm(prototype2.t())
    return torch.diagonal(similarity,0)


class soft_ContrastiveLoss(torch.nn.Module):

    def __init__(self, alpha=0.25, gamma=1, margin=0.2):
        super(soft_ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sigmoid =  nn.Sigmoid()
        self.alpha=alpha
        self.gamma=gamma
        self.EPS = 1e-12
        self.soft_plus = nn.Softplus ()


    def forward(self, output1, output2, label):  ##similarity
        similarity = cosine_distance(output1, output2)
        # weight_n=(self.margin+similarity)##(self.margin+similarity)**self.gamma
        # weight_p=(1+self.margin-similarity)##**self.gamma
        weight_n=torch.clamp_min((self.margin+similarity), min=0.)
        weight_p=torch.clamp_min((1+self.margin-similarity), min=0.)
        neg=torch.exp ((label)*weight_n * (similarity -self.margin) * self.gamma)
        neg_sum= torch.sum(neg)
        pos=torch.exp ((1-label)*weight_p * ((1-self.margin)-similarity) * self.gamma)
        pos_sum = torch.sum(pos)
        loss = self.soft_plus(neg_sum + pos_sum)
        return loss/output1.size(0)

    def forward_v1(self, output1, output2, label):  ##similarity  class imbalance
        similarity = cosine_distance (output1, output2)
        # weight_n=(self.margin+similarity)##(self.margin+similarity)**self.gamma
        # weight_p=(1+self.margin-similarity)##**self.gamma
        weight_n = (1-self.alpha)*torch.clamp_min ((self.margin + similarity), min=0.)
        weight_p = self.alpha*torch.clamp_min ((1 + self.margin - similarity), min=0.)
        neg = torch.exp ((label) * weight_n * (similarity - self.margin) * self.gamma)
        neg_sum = torch.sum (neg)
        pos = torch.exp ((1 - label) * weight_p * ((1 - self.margin) - similarity) * self.gamma)
        pos_sum = torch.sum (pos)
        loss = self.soft_plus (neg_sum + pos_sum)
        return loss / output1.size (0)
    def change_margin(self, T, T_max):
        a = generate_margin(T, T_max)
        self.margin = a
        print("change margin = ", a)


def generate_margin(T, T_max):
    # a = 2*(1-(math.pow(float((T/T_max)),2)))
    a = 2+ 4* ((math.pow (float ((T / T_max)), 2)))
    # a = 2 * (1 - (math.pow (float ((T / T_max)), 2)))
    return a


if __name__ == '__main__':
    loss = soft_ContrastiveLoss()
    a = torch.rand((8,5))
    b = torch.rand((8,5))
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    label = torch.Tensor([1,0,1,0,1,0,1,0])
    loss1 = loss(a,b,label)
    print(loss1)
