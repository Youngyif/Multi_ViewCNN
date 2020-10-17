import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.5, gamma=0.5):
        super(FocalLoss,self).__init__()
        self.alpha=alpha
        self.gamma=gamma
        self.EPS=1e-12

    def forward(self, input, target):
            #
            # pt=torch.randn(input.size())
            # pt=Variable(pt)
            pt=input.view(-1)*(target.view(-1)==1.).float()+(1-input.view(-1))*(target.view(-1)==0.).float()
#         loss = -self.alpha * (torch.pow((1 - pt), self.gamma)) * torch.log(pt + self.EPS)
            loss=-self.alpha*(torch.pow((1-pt),self.gamma))*torch.log(pt+self.EPS)*(target.view(-1)==1.).float()-\
            (1 - self.alpha)* (torch.pow((1 - pt), self.gamma)) * torch.log(pt + self.EPS) * (target.view(-1) == 0.).float()
            self.loss = loss.mean()
            return self.loss


class FocalLoss(nn.Module):
    def  __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)

        # print(">>>!!!", target)
        logpt = logpt.gather(1,target)
        # print(">>>!!!",logpt)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()