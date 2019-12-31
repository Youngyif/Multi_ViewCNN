import os
import math


class NetOption (object):

    def __init__(self):
        #  ------------ General options ---------------------------------------
        self.save_path = ""  # log path
        self.data_path = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/" # path for loading data set  \
        self.label_path = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/3dlabel_only_narrow.csv"
        self.rootpath = "/mnt/dataset/splited_Casia2"
        self.data_set = "asoct"  # options: asoct
        self.disease_type = 1  # 1(open) | 2(narrow) | 3(close) | 4(unclassify)  or  1(open) | 2(narrow/close)
        self.manualSeed = 1  # manually set RNG seed
        self.nGPU = 1  # number of GPUs to use by default
        self.GPU = 2  # default gpu to use, options: range(nGPU)
        self.datasetRatio = 1.0  # greedy increasing training data for cifar10

        # ------------- Data options ------------------------------------------
        self.nThreads = 10  # number of data loader threads

        # ------------- Training options --------------------------------------
        self.testOnly = True  # run on validation set only
        self.tenCrop = False  # Ten-crop testing

        # ---------- Optimization options -------------------------------------
        self.nEpochs = 80  # number of total epochs to train
        self.batchSize = 16  # mini-batch size
        self.LR = 0.001  # initial learning rate
        self.lrPolicy = "multistep"  # options: multistep | linear | exp
        self.momentum = 0.9  # momentum
        self.weightDecay = 1e-4  # weight decay 1e-2
        self.gamma = 0.94  # gamma for learning rate policy (step)
        self.step = 2.0  # step for learning rate policy

        # ---------- Model options --------------------------------------------
        self.trainingType = 'onevsall'  # options: onevsall | multiclass
        self.netType = "multi_viewCNN"  # options: ResNet | DenseNet | Inception-v3 | AlexNet
        self.experimentID = "darkvslight_1104_onlyNarrow"
        self.depth = 18  # resnet depth: (n-2)%6==0
        self.wideFactor = 1  # wide factor for wide-resnet
        self.numOfView = 21
        # ---------- Resume or Retrain options --------------------------------
        ##v3

        # self.model1 = '/home/yangyifan/jingwen_code_oct_v2/as-oct/log_asoct_ResNet_50_onevsall_bs16_type=2-lr=0.001/model/best_model.pkl'## load model for output
        # self.retrain = [self.model1]  # path to model to retrain with
        self.retrain = None  # path to model to retrain with
        self.resume = None  # path to directory containing checkpoint
        # self.resume = "/usr/home2/code/jingwen_code_oct_cropped/as-oct/log_asoct_ResNet_18_onevsall_bs8_addpad_9.6_3foldType=1-lr=0.01/model/checkpoint7.pkl"
        self.resumeEpoch = 0  # manual epoch number for resume
        self.pretrain = 'pretrain/resnet18.pth'

        # check parameters
        self.paramscheck ()

    def paramscheck(self):
        self.save_path = "log_%s_%s_%d_%s_bs%d_%s/" % \
                         (self.data_set, self.netType,
                          self.depth, self.trainingType, self.batchSize, self.experimentID)
        if self.data_set == 'asoct':
            if self.trainingType == 'onevsall':
                self.nClasses = 1
            else:
                self.nClasses = 4
            self.ratio = [1.0 / 2, 2.7 / 3]
