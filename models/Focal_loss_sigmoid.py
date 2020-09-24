import torch, sys, os, pdb
import torch.nn as nn
import torch.nn.functional as F

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=True):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.logits = logits
#         self.reduce = reduce
#
#     def forward(self, inputs, targets):
#         if self.logits:
#             BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
#         else:
#             BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
#
#         if self.reduce:
#             return torch.mean(F_loss)
#         else:
#             return F_loss

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.75, gamma=2):
        super(FocalLoss,self).__init__()
        self.alpha=alpha
        self.gamma=gamma
        self.EPS=1e-12

    def forward(self, input, target):
            #
            # pt=torch.randn(input.size())
            # pt=Variable(pt)
            pt=input.view(-1)*(target.view(-1)==1.).float()+(1-input.view(-1))*(target.view(-1)==0.).float()
#         loss = -self.alpha * (torch.pow((1 - pt), self.gamma)) * torch.log(pt + self.EPS)
            loss=-self.alpha*(torch.pow((1-pt),self.gamma))*torch.log(pt+self.EPS)*(target.view(-1)==1.).float()-\
            (1 - self.alpha)* (torch.pow((1 - pt), self.gamma)) * torch.log(pt + self.EPS) * (target.view(-1) == 0.).float()
            self.loss = loss.mean()
            return self.loss