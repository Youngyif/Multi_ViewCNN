import os
import math


class NetOption (object):

    def __init__(self):
        #  ------------ General options ---------------------------------------
        self.save_head_path = "/home/yangyifan/model/synechiae/"  # where to save model and log code etc#/home/yangyifan/model/synechiae/#/mnt/dataset/model/darklight/
        self.data_path = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/" # path for loading data set  \
        # self.label_path = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/oneclock_data_split/3dlabel_open_narrow.csv"
        # self.label_path = "/home/yangyifan/code/multiviewCNN_quarter/multiViewCNN/Multi_ViewCNN/dataProcess/oneclockv4_open_narrow18.csv"###0330
        self.label_path ="/home/yangyifan/code/multiviewCNN_quarter/multiViewCNN/Multi_ViewCNN/dataProcess/add_alone_clock/one_for_new_clock_labelv1.csv"###one lable cut in half 0413
        # self.label_path = "/home/yangyifan/code/multiviewCNN_quarter/multiViewCNN/Multi_ViewCNN/dataProcess/add_alone_clock/one_for_new_clock_labelv2_0428.csv"###one label like clock 0428
        # self.label_path ="/home/yangyifan/code/multiviewCNN_quarter/multiViewCNN/Multi_ViewCNN/dataProcess/external_label_process/medical_label_external/RU_one_cutInHalf_label.csv"##RU ONE LABEL
        # self.label_path = "/home/yangyifan/code/multiviewCNN_quarter/multiViewCNN/Multi_ViewCNN/dataProcess/external_label_process/medical_label_external/RU_half_label.csv"  ##RU HALF LABEL
        # self.label_path = "/home/yangyifan/code/multiViewCNN/multi-viewCNN/dataProcess/label_version/v2_exisit_noaloneclock_half_3d_label.csv"
        # self.label_path = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/wide_split/quartersplit/label_quarter.csv"
        self.test_log = "/home/yangyifan/model/synechiae/testlogs/testlog.txt"
        # self.label_path = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/wide_split/label for all/label_half_opennarrow.csv" ##half label
        # self.label_path ="/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/External_set/overlapmethod/RU_9798Label_badcase.csv"
        self.rootpath = "/mnt/dataset/splited_Casia2"
        # self.rootpath = "/home/yangyifan/dataset/Splited_External_Casia/CASIAII-RU_resize"
        self.data_set = "asoct"  # options: asoct
        self.disease_type = 1  # 1(open) | 2(narrow) | 3(close) | 4(unclassify)  or  1(open) | 2(narrow/close)
        self.manualSeed = 1  # manually set RNG seed
        self.nGPU = 2  # number of GPUs to use by default
        self.GPU =4 # default gpu to use, options: range(nGPU)
        self.datasetRatio = 1.0  # greedyincreasing training data for cifar10
        self.numclass = 1
        # ------------- Data options ------------------------------------------
        self.nThreads = 10  # number of data loader threads

        # ------------- Training options --------------------------------------
        self.testOnly = False  # run on validation set only
        self.tenCrop = False  # Ten-crop testing

        # ---------- Optimization options -------------------------------------
        self.nEpochs = 200  # number of total epochs to train
        self.batchSize = 8  # mini-batch size
        self.LR = 0.001  # initial learning rate
        self.lrPolicy = "multistep"  # options: multistep | linear | exp
        self.momentum = 0.9  # momentum
        self.weightDecay = 1e-4  # weight decay 1e-2
        self.gamma = 0.94  # gamma for learning rate policy (step)
        self.step = 2.0  # step for learning rate policy

        # ---------- Model options --------------------------------------------
        self.trainingType = 'onevsall'  # options: onevsall | multiclass
        self.netType = "I3D"  # options: ResNet | DenseNet | Inception-v3 | AlexNet |resnet3d |multi_viewCNN |lstm_mvcnn |dual_resnet3d|dual_extract_resnet3d | C3D | I3D  | S3D | slowfast
        self.experimentID = "multiscale_ratio0.1_resnet3d_cumulative_CONTRA0.5-0_oversample_alpha0.75_1008"
        self.depth = 18  # resnet depth: (n-2)%6==0
        self.wideFactor = 1  # wide factor for wide-resnet
        self.numOfView = 10
        # self.cat = True

        self.cat = False
        self.mscale = False
        self.structure = False
        self.attention = False
        self.contra = False
        self.contra_focal = True
        self.twoway = False
        self.contra_single = False
        self.contra_learning = False
        self.contra_learning_2 = False
        self.contra_multiscale = True
        self.imgsize = 244  ##112 FOR C3D  224 FOR I3D 244 FOR S3D
        self.typedata = "both"  ##dark|light|both
        ###--draw Roc---###
        self.draw_ROC = False
        # ---------- Resume or Retrain options --------------------------------
        ##v3

        # self.retrain = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/pretrain/checkpoint18.pkl"## load model for output
        # self.retrain = [self.model1]  # path to model to retsrain with
        self.retrain = None  # path to model to retrain with
        # self.resume = "/mnt/dataset/model/darklight/log_asoct_resnet3d_18_onevsall_bs8_focal_balance_alpha0.75_sigmoid_SORT_FN_INDEX_permute_newone_light_resnet3d_0421/model/checkpoint11.pkl" # path to directory containing checkpoint
        # self.resume ="/mnt/dataset/model/darklight/log_asoct_resnet3d_18_onevsall_bs8_shuffle_index0_oversample__focal_balance_alpha0.75_sigmoid_SORT_FN_INDEX_permute_newone_light_resnet3d_0424/model/best_model.pkl"
        # self.resume ="/mnt/dataset/model/darklight/log_asoct_resnet3d_18_onevsall_bs8_shuffle_index0_oversample__focal_balance_alpha0.75_sigmoid_SORT_FN_INDEX_permute_newone_contra_resnet3d_0424/model/best_model.pkl"
        # self.resume = "/home/yangyifan/model/synechiae/log_asoct_resnet3d_18_onevsall_bs8_CutInHalf_multiscale_resnet3d_cumulative_CONTRA0.5-0_oversample_alpha0.75_0930/model/checkpoint88.pkl"
        self.resume = None
        # self.resume_I3D = "/home/yangyifan/model/synechiae/log_asoct_I3D_18_onevsall_bs8_CutInHalf_baseline_I3DCONTRA0.1_oversample_alpha0.75_0914/model/best_model.pkl"
        # self.resume_I3D = "/home/yangyifan/model/synechiae/log_asoct_I3D_18_onevsall_bs8_baseline_contra1_I3D_oversample_alpha0.75_0723/model/checkpoint77.pkl"
        # self.test_log = "/home/yangyifan/model/test_log/log_asoct_I3D_18_onevsall_bs8_baseline_contra1_I3D_oversample_alpha0.75_0723_checkpoint77.txt"
        """
        "/home/yangyifan/model/synechiae/log_asoct_I3D_18_onevsall_bs8_baseline_contra1_I3D_oversample_alpha0.75_0723/model/checkpoint77.pkl"
        
        """
        self.resumeEpoch = 0  # manual epoch number for resume
        # self.retrain = "/mnt/dataset/model/darklight/log_asoct_Single_viewCNN_18_onevsall_bs16_half_mvcmm_twostage_pretrain_0213/model/best_model.pkl"
        # self.pretrain = None
        self.pretrain = "/home/yangyifan/code/pytorch-resnet3d/pretrained/i3d_r50_nl_kinetics.pth"
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
