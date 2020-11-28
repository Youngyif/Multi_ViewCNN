import torch
import torch.nn.functional as F
import torch.nn as nn
import math


def cosine_distance(prototype1, prototype2):
    # prototype1 = F.normalize(prototype1.unsqueeze(0))
    # prototype2 = F.normalize(prototype2.unsqueeze(0))
    similarity = prototype1.mm(prototype2.t())
    return 1 - similarity


class Focal_ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, save_hyper, alpha=0.25, gamma=2, margin=2, scale=2):
        super(Focal_ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sigmoid = nn.Sigmoid()
        self.alpha=alpha
        self.gamma=gamma
        self.EPS = 1e-12
        self.scale=scale
        self.savehyper = save_hyper
        self.savehyper.write_to_dict ("margin_of_focal_contrastive", margin)
        self.savehyper.write_to_dict ("gamma_of_focal_contrastive", gamma)
        self.savehyper.write_to_dict ("alpha_of_focal_contrastive", alpha)
        self.savehyper.write_to_dict ("scale_of_focal_contrastive", scale)


    def forward(self, output1, output2, label): ##pairwise distance
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        euclidean_distance=euclidean_distance.squeeze(1)
        # print("euclidean", euclidean_distance)
        pt = self.sigmoid (euclidean_distance)# pt 越大则 相似度越小   y=1代表不相似  y=0代表相似 相似的时候pt越大代表越难   不相似的时候pt越小则越难
        dis_sim=(1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        sim = (label) * torch.pow (euclidean_distance, 2)
        # print("label",label)
        # print("pow", torch.pow (euclidean_distance, 2))
        # print ("sim", sim)
        # mine_dis_sim = dis_sim[dis_sim <max(sim)]
        # mine_sim=sim[sim>min(dis_sim)]
        # print ("sim", mine_sim)
        # print(label)

        # print("sim",sim)
        dis_sim = (self.alpha)*self.scale*dis_sim*torch.pow((1-pt),self.gamma)
        sim = (1-self.alpha)*self.scale*sim*torch.pow((pt+self.EPS), self.gamma)
        # print ("sim alpha", sim)
        loss_contrastive = torch.mean(dis_sim+sim)

        # loss_contrastive = torch.mean(self.scale*(torch.pow((pt+self.EPS), self.gamma))*(1-label) * torch.pow(euclidean_distance, 2) +\
        #                               self.scale*(torch.pow((1-pt),self.gamma))*(label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

    # def forward_v2(self, output1, output2, label):  ##similarity
    #     euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
    #     pt = self.sigmoid(euclidean_distance)  #torch.mean()
    #     # loss_contrastive = (self.alpha)*(torch.pow((pt+self.EPS), self.gamma))*(1-label) * torch.pow(euclidean_distance, 2) +\
    #     #                                            (1-self.alpha)*(torch.pow((1-pt),self.gamma))*(label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
    #     sim=(torch.pow((pt+self.EPS), self.gamma))*(1-label) * torch.pow(euclidean_distance, 2)
    #     dis_sim=(torch.pow((1-pt),self.gamma))*(label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
    #     loss_contrastive = torch.mean(sim+dis_sim)
    #
    #
    #     return loss_contrastive
    #
    # def forward_v3(self, output1, output2, label):  ##similarity
    #     # euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
    #     similarity = cosine_distance(output1, output2)
    #     pt =similarity/2
    #     # pt = self.sigmoid(euclidean_distance)  #torch.mean()
    #     loss_contrastive = torch.mean((torch.pow((pt+self.EPS), self.gamma))*(1-label) * torch.pow(similarity, 2) +\
    #                                                2*(torch.pow((1-pt),self.gamma))*(label) * torch.pow(torch.clamp(self.margin - similarity, min=0.0), 2))
    #
    #
    #     return loss_contrastive


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
    loss = Focal_ContrastiveLoss()
    a = torch.rand((8,5))
    b = torch.rand((8,5))
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    label = torch.ones(8,1)
    loss1 = loss(a,b,label)
    print(loss1)
