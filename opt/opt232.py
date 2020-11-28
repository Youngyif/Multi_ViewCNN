import os
import math


class NetOption (object):

    def __init__(self):
        #  ------------ General options ---------------------------------------
        self.save_head_path = "/mnt/cephfs/home/yangyifan/yangyifan/232/yangyifan/232model/darklight/"  # where to save model and log code etc
        self.data_path = "/mnt/cephfs/home/yangyifan/yangyifan/232/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/" # path for loading data set  \
        # self.label_path = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/3dlabel_only_narrow.csv"
        # self.label_path = "/home/yangyifan/code/multiViewCNN/multi-viewCNN/dataProcess/label_version/v2_exisit_noaloneclock_half_3d_label.csv"
        # self.label_path = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/wide_split/quartersplit/label_quarter.csv"
        # self.label_path = "/home/yangyifan/code/multi_view_0812/dataProcess/one_for_new_clock_labelv1.csv"#internal label 0826
        self.label_path = "/mnt/cephfs/home/yangyifan/yangyifan/232/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/split_bjdata/BJ_dl_label.csv" # BJ label 0826
        # self.label_path = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/wide_split/label for all/label_half_opennarrow.csv"
        self.rootpath = "/mnt/cephfs/home/yangyifan/yangyifan/232/yangyifan/232data/splited_bj_resize/BJ_Resizev1/"#"/home/datasets/CASIA2/splited_Casia2/"
        self.data_set = "asoct"  # options: asoct
        self.disease_type = 1  # 1(open) | 2(narrow) | 3(close) | 4(unclassify)  or  1(open) | 2(narrow/close)
        self.manualSeed = 1  # manually set RNG seed
        self.nGPU = 2  # number of GPUs to use by default
        self.GPU =6# default gpu to use, options: range(nGPU)
        self.datasetRatio = 1.0  # greedy increasing training data for cifar10
        self.numclass = 1
        # ------------- Data options ------------------------------------------
        self.nThreads = 10  # number of data loader threads
        self.dataset = "BJ"   # BJ | internal
        # self.typedata = "dark" #dark | light
        self.imgsize = 244
        # ------------- Training options --------------------------------------
        self.testOnly = False  # run on validation set only
        self.tenCrop = False  # Ten-crop testing

        # ---------- Optimization options -------------------------------------
        self.nEpochs = 200  # number of total epochs to train
        self.batchSize = 8 # mini-batch size
        self.LR = 0.001  # initial learning rate
        self.lrPolicy = "multistep"  # options: multistep | linear | exp
        self.momentum = 0.9  # momentum
        self.weightDecay = 1e-4  # weight decay 1e-2
        self.gamma = 0.94  # gamma for learning rate policy (step)
        self.step = 2.0  # step for learning rate policy

        # ---------- Model options --------------------------------------------
        # self.cat = True
        self.cat = False
        self.mscale = False
        self.structure = False
        self.attention = False
        self.contra = False
        self.contra_focal = True
        self.multiway_contra = False
        self.circle_loss = False
        self.contra_focal_bilinear = False
        self.contra_single = False
        self.contra_learning = False
        self.contra_learning_2 = False
        self.contra_multiscale = True
        ###--draw Roc---###
        self.draw_ROC = False
        # ---------- Model options --------------------------------------------
        self.trainingType = 'onevsall'  # options: onevsall | multiclass
        self.netType = "resnet3d"  # options: | C3D | I3D  | S3D | slowfast | resnet3d | multi_viewCNN |lstm_mvcnn |dual_resnet3d|dual_extract_resnet3d | TSN
        self.experimentID = "squeeze_resnet3d_alpha0.25_margin2_featuredim=21_addpositivemargin_scale=2_gamma2_focal_#real#contra_reduce=1_1128"  ##"resnet3d_multiway_CONTRA_MARGIN=2_RATIO=0.1_pretrain_0917"
        self.depth = 18  # resnet depth: (n-2)%6==0
        self.wideFactor = 1  # wide factor for wide-resnet
        self.numOfView = 10
        self.margin = 1.5
        self.loss_ratio = 0.1

        ##don't fuse
        self.multifuse = False
        self.singleinput = False
        ##
        ## mixup
        self.mixup = False
        self.alpha = 0.1
        ##mix up
        # ---------- Resume or Retrain options --------------------------------
        ##v3
        self.resume_I3D = False
        # self.retrain = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/pretrain/checkpoint18.pkl"## load model for output
        # self.retrain = [self.model1]  # path to model to retrain with
        self.retrain = None  # path to model to retrain with
        self.resume = None  # path to directory containing checkpoint
        # self.resume = "/usr/home2/code/jingwen_code_oct_cropped/as-oct/log_asoct_ResNet_18_onevsall_bs8_addpad_9.6_3foldType=1-lr=0.01/model/checkpoint7.pkl"
        # self.resume = "/home/yangyifan/code/multi_view_0812/resume/contra_best_model.pkl"#contra
        # self.resume = "/home/yangyifan/code/multi_view_0812/resume/dark_best_model.pkl"#dark
        # self.resume = "/home/yangyifan/code/multi_view_0812/resume/light_best_model.pkl"
        # self.resume = "/home/datasets/CASIA2/model/darklight/log_asoct_resnet3d_18_onevsall_bs60_TEST_TIME_dark_resnet3d_0812/model/best_model.pkl"
        # self.resume = "/home/datasets/CASIA2/model/darklight/log_asoct_resnet3d_18_onevsall_bs60_TEST_TIME_LIGHT_resnet3d_0812/model/best_model.pkl"
        # self.resume = "/home/datasets/CASIA2/model/darklight/log_asoct_resnet3d_18_onevsall_bs32_TEST_TIME_CONTRA_resnet3d_0812/model/best_model.pkl"
        self.resumeEpoch = 0  # manual epoch number for resume
        # self.retrain = "/mnt/dataset/model/darklight/log_asoct_Single_viewCNN_18_onevsall_bs16_half_mvcmm_twostage_pretrain_0213/model/best_model.pkl"
        self.pretrain = None
        self.resume=None
        # self.resume = "/home/datasets/CASIA2/model/darklight/log_asoct_resnet3d_18_onevsall_bs8_resnet3d_contra_multiscale_ratio_fix_ratio=1_no_pretrain_large_mudule0.5_1012/model/best_model.pkl"
        self.pretrain = "/mnt/cephfs/home/yangyifan/yangyifan/232/yangyifan/code/multiViewCNN/pretrained/i3d_r50_nl_kinetics.pth"
        # self.pretrain = "/mnt/dataset/model/darklight/log_asoct_dual_resnet3d_18_onevsall_bs4_half_opennarrow_fixBeforeFc_0219/model/best_model.pkl"
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
