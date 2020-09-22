# import os
# import math
#
#
# class NetOption (object):
#
#     def __init__(self):
#         #  ------------ General options ---------------------------------------
#         self.save_head_path = "/mnt/dataset/model/darklight/"  # where to save model and log code etc
#         self.data_path = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/" # path for loading data set  \
#         # self.label_path = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/oneclock_data_split/3dlabel_open_narrow.csv"
#         # self.label_path = "/home/yangyifan/code/multiviewCNN_quarter/multiViewCNN/Multi_ViewCNN/dataProcess/oneclockv4_open_narrow18.csv"###0330
#         self.label_path ="/home/yangyifan/code/multiviewCNN_quarter/multiViewCNN/Multi_ViewCNN/dataProcess/add_alone_clock/one_for_new_clock_labelv1.csv"
#         # self.label_path = "/home/yangyifan/code/multiViewCNN/multi-viewCNN/dataProcess/label_version/v2_exisit_noaloneclock_half_3d_label.csv"
#         # self.label_path = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/wide_split/quartersplit/label_quarter.csv"
#         # self.label_path = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/wide_split/label for all/label_half_opennarrow.csv"
#         self.rootpath = "/mnt/dataset/splited_Casia2"
#         self.data_set = "asoct"  # options: asoct
#         self.disease_type = 1  # 1(open) | 2(narrow) | 3(close) | 4(unclassify)  or  1(open) | 2(narrow/close)
#         self.manualSeed = 1  # manually set RNG seed
#         self.nGPU = 2  # number of GPUs to use by default
#         self.GPU = 6# default gpu to use, options: range(nGPU)
#         self.datasetRatio = 1.0  # greedyincreasing training data for cifar10
#         self.numclass = 1
#         # ------------- Data options ------------------------------------------
#         self.nThreads = 10  # number of data loader threads
#
#         # ------------- Training options --------------------------------------
#         self.testOnly = False  # run on validation set only
#         self.tenCrop = False  # Ten-crop testing
#
#         # ---------- Optimization options -------------------------------------
#         self.nEpochs = 200  # number of total epochs to train
#         self.batchSize = 8  # mini-batch size
#         self.LR = 0.001  # initial learning rate
#         self.lrPolicy = "multistep"  # options: multistep | linear | exp
#         self.momentum = 0.9  # momentum
#         self.weightDecay = 1e-4  # weight decay 1e-2
#         self.gamma = 0.94  # gamma for learning rate policy (step)
#         self.step = 2.0  # step for learning rate policy
#
#         # ---------- Model options --------------------------------------------
#         self.trainingType = 'onevsall'  # options: onevsall | multiclass
#         self.netType = "resnet3d"  # options: ResNet | DenseNet | Inception-v3 | AlexNet |resnet3d |multi_viewCNN |lstm_mvcnn |dual_resnet3d|dual_extract_resnet3d
#         self.experimentID = "test_permute_newone_contra_resnet3d_0401"
#         self.depth = 18  # resnet depth: (n-2)%6==0
#         self.wideFactor = 1  # wide factor for wide-resnet
#         self.numOfView = 10
#         # self.cat = True
#
#         self.cat = False
#         self.mscale = False
#         self.structure = False
#         self.attention = False
#         self.contra = True
#         ###--draw Roc---###
#         self.draw_ROC = False
#         # ---------- Resume or Retrain options --------------------------------
#         ##v3
#
#         # self.retrain = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/pretrain/checkpoint18.pkl"## load model for output
#         # self.retrain = [self.model1]  # path to model to retsrain with
#         self.retrain = None  # path to model to retrain with
#         # self.resume = "/mnt/dataset/model/darklight/log_asoct_resnet3d_18_onevsall_bs4_multiscale_one_light_resnet3d_0324/model/best_model.pkl" # path to directory containing checkpoint
#         self.resume = None
#         self.resumeEpoch = 0  # manual epoch number for resume
#         # self.retrain = "/mnt/dataset/model/darklight/log_asoct_Single_viewCNN_18_onevsall_bs16_half_mvcmm_twostage_pretrain_0213/model/best_model.pkl"
#         # self.pretrain = None
#         self.pretrain = "/home/yangyifan/code/pytorch-resnet3d/pretrained/i3d_r50_nl_kinetics.pth"
#         # self.pretrain = "/mnt/dataset/model/darklight/log_asoct_dual_resnet3d_18_onevsall_bs4_half_opennarrow_fixBeforeFc_0219/model/best_model.pkl"
#         # check parameters
#         self.paramscheck ()
#
#     def paramscheck(self):
#         self.save_path = "log_%s_%s_%d_%s_bs%d_%s/" % \
#                          (self.data_set, self.netType,
#                           self.depth, self.trainingType, self.batchSize, self.experimentID)
#         if self.data_set == 'asoct':
#             if self.trainingType == 'onevsall':
#                 self.nClasses = 1
#             else:
#                 self.nClasses = 4
#             self.ratio = [1.0 / 2, 2.7 / 3]
