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
        self.savehyper = save_hyper
        self.savehyper.write_to_dict ("margin_of_focal_contrastive", margin)
        self.margin_mine=0.8
        # self.savehyper.write_to_dict ("gamma_of_focal_contrastive", gamma)
        # self.savehyper.write_to_dict ("alpha_of_focal_contrastive", alpha)

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        label1=label
        # print("before squeeze",euclidean_distance)
        euclidean_distance = euclidean_distance.squeeze (1)
        # print("after squeeze",euclidean_distance)
        # a=(1-label) * torch.pow(euclidean_distance-self.margin, 2)
        # print("a",a)
        sim_distance = euclidean_distance[label == 1]
        dis_sim_distance = euclidean_distance[label == 0]
        if len (sim_distance) > 0:
            mine_dis_sim = dis_sim_distance[dis_sim_distance - self.margin_mine < max (sim_distance)]  ##0.4 mining stategy
        else:
            mine_dis_sim = dis_sim_distance
        # print("mine_dis", mine_dis_sim)
        # mine_sim = sim_distance[sim_distance > min (dis_sim_distance)]
        mine_sim = sim_distance
        loss_contrastive = (torch.sum(torch.pow(mine_sim, 2))+torch.sum(torch.pow(torch.clamp(self.margin - mine_dis_sim, min=0.0), 2)))/(int(mine_sim.size(0)+mine_dis_sim.size(0)))



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
    a = torch.randn(8,5)
    b = torch.randn(8,5)
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    labels=torch.Tensor([1,0,1,0,1,0,0,1])
    loss1 = loss(a,b,labels)
    print(loss1)
