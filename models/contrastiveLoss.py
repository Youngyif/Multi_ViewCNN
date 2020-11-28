import torch
import torch.nn.functional as F
import torch.nn as nn
import math
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, save_hyper=None ,margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sigmoid =  nn.Sigmoid()
        # self.savehyper = save_hyper
        # self.savehyper.write_to_dict ("margin_of_focal_contrastive", margin)
        self.margin_sim=0.2
        # self.savehyper.write_to_dict ("gamma_of_focal_contrastive", gamma)
        # self.savehyper.write_to_dict ("alpha_of_focal_contrastive", alpha)

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance-self.margin, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive




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
    loss = ContrastiveLoss()
    a = torch.randn(64,5)
    b = torch.randn(64,5)
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    loss1 = loss(a,b,1)
    print(loss1)
