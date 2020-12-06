import torch
import torchvision.transforms as transforms
# from dataloader.myloader_multiscale import  Myloader
from PIL import Image
from dataloader.my_loader_oneclock_newquarter_old_version import *  ###cut in half
# from dataloader.myload_step_one_internal import * ###one clock like clock 21 images 0429
# from dataloader.myloader_BJ import *
# from dataloader.myloader_step2_16 import *
# from dataloader.myloader_step9_internal import *
# from dataloader.myloader_RU import *
# from dataloader.myloader16 import *
# from dataloader.overlap_ru_bidirection import *
# from dataloader.overlap_ru_one import *
# from dataloader.overlap_loader import *
# from dataloader.myloader_RU_half import * ##half clock in RU
# from dataloader.myloader_half import * ##half clock in INTERNAL
import numpy as np
import random
from trainer import *
from dataloader.torchsampler import ImbalancedDatasetSampler
from torch.utils.data.sampler import  WeightedRandomSampler

class RandomGammaCorrection():
    def __init__(self):
        self.gamma = 1.0

    def __call__(self, img):
        img = np.asarray(img)
        img = np.power(img / 255.0, self.gamma)
        img = np.uint8(img * 255.0)

        return Image.fromarray(img)


    def randomize_parameters(self, custom_extend=None):
        self.gamma = np.random.uniform(1, 2, 1)
        if random.random() < 0.5:
            self.gamma = 1 / self.gamma

class VerticalCrop(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.vertical_center = None
        self.ratio = 0.05
        self.hwr = 1.2

    def __call__(self, img):
        image_width = img.size[0]
        image_height = img.size[1]
        vertical_center = self.vertical_center + int(self.ratio * image_width)
        crop_height = image_width * self.hwr
        if vertical_center - crop_height // 2 < 0:
            x1 = 0
            y1 = 0
            x2 = image_width
            y2 = crop_height
        elif vertical_center + crop_height // 2 > image_height:
            x1 = 0
            y1 = image_height - crop_height
            x2 = image_width
            y2 = image_height
        else:
            x1 = 0
            y1 = vertical_center - crop_height // 2
            x2 = image_width
            y2 = vertical_center + crop_height // 2

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self, vertical_center):
        """
        custom_extend: vertical center coordinate for estimated spur sceleral position.
        """
        self.vertical_center = vertical_center

class RandomVerticalCrop(object):

    def __init__(self, size,interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.vertical_center = None
        self.ratio = 0.05
        self.hwr = 1.2

    def __call__(self, img):
        image_width = img.size[0]
        image_height = img.size[1]
        vertical_center = self.vertical_center + int(self.ratio * image_width)
        crop_height = image_width * self.hwr
        if vertical_center - crop_height // 2 < 0:
            x1 = 0
            y1 = 0
            x2 = image_width
            y2 = crop_height
        elif vertical_center + crop_height // 2 > image_height:
            x1 = 0
            y1 = image_height - crop_height
            x2 = image_width
            y2 = image_height
        else:
            x1 = 0
            y1 = vertical_center - crop_height // 2
            x2 = image_width
            y2 = vertical_center + crop_height // 2

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size, self.size), self.interpolation)

class DataLoader (object):
    def __init__(self, dataset, data_path, label_path, batch_size, rootpath, n_threads=4, ten_crop=False, dataset_ratio=1.0):
        self.dataset = dataset
        self.data_path = data_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.ten_crop = ten_crop
        self.rootpath = rootpath
        if self.dataset == "asoct":
            print ("|===>Creating %s Data Loader" % (self.dataset))
            self.train_loader, self.test_loader = self.asoct_data (data_path=self.data_path, label_path=self.label_path, rootpath = self.rootpath)
        # elif self.dataset == "validation":
        #     print ("|===>Creating %s Data Loader" % (self.dataset))
        #     # self.val_loader = self.palmdata(dataset=self.dataset, data_path=self.data_path)
        #     self.val_loader = self.asoct_data_val (data_path=self.data_path, label_path=self.label_path, rootpath = self.rootpath)
        else:
            assert False, "invalid data set"

    def getloader(self):
        if self.dataset == "validation":
            return self.val_loader
        else:
            return self.train_loader, self.test_loader

    # def asoct_data_val(self, data_path, label_path, rootpath):
    #     imgSize = 244
    #     # test_dir = data_path + "test.txt"
    #     test_dir = data_path + "trainval.txt"
    #
    #     normalize = transforms.Normalize(mean=[0.146, 0.146, 0.146],
    #                                      std=[0.193, 0.193, 0.193])
    #
    #     test_transform = transforms.Compose ([
    #         # transforms.Scale(600),
    #         transforms.Scale (244),
    #         transforms.CenterCrop (imgSize),
    #         transforms.ToTensor (),
    #         normalize,
    #     ])
    #     test_loader = torch.utils.data.DataLoader (
    #         Myloader (rootpath, test_dir, label_path, test_transform),
    #         batch_size=self.batch_size,
    #         shuffle=True,
    #         num_workers=self.n_threads,
    #         pin_memory=False)
    #     return test_loader

    def asoct_data(self, data_path, label_path, rootpath):
        imgSize = opt.imgsize
        if opt.dataset=="internal":
            train_dir = "/mnt/cephfs/home/yangyifan/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/oneclock_data_split/oversampling/overtrain_shuffle.txt"##fix version 0430
            test_dir = "/mnt/cephfs/home/yangyifan/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/oneclock_data_split/valv5.txt"##0413##fix version 0430
        if opt.dataset == "BJ":
            train_dir = "/mnt/cephfs/home/yangyifan/yangyifan/232/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/split_bjdata/trainvallist/train_bj.txt" ##BJ TRAIN
            # train_dir = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/split_bjdata/trainvallist/overtrain/train_bj.txt"
            test_dir = "/mnt/cephfs/home/yangyifan/yangyifan/232/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/split_bjdata/trainvallist/test_bj.txt"  ##BJ TEST

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        imagenet_pca = {
            'eigval': torch.Tensor ([0.2175, 0.0188, 0.0045]),
            'eigvec': torch.Tensor ([
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ])
        }
        train_loader = torch.utils.data.DataLoader (
            Myloader (rootpath, train_dir, label_path, transforms.Compose ([
                # transforms.Scale(600),
                # transforms.Grayscale (num_output_channels=1), ##转为灰度图 output channel 由3改为1
                transforms.Scale (imgSize+5),
                # transforms.RandomHorizontalFlip (),
                # transforms.RandomVerticalFlip(),
                # transforms.CenterCrop(imgSize),
                transforms.RandomCrop(imgSize),
                transforms.RandomRotation(10),
                transforms.ToTensor (),
                # auxtransform.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                # auxtransform.Lighting(alphastd=0.1, eigval=imagenet_pca['eigval'], eigvec=imagenet_pca['eigvec']),
                normalize,
            ])),
            # sampler=WeightedRandomSampler(Myloader),#imbalanced data
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_threads,
            pin_memory=True)

        test_transform = transforms.Compose ([
            # transforms.Grayscale(num_output_channels=1),  ##转为灰度图 output channel 由3改为1
            # transforms.Scale(600),
            transforms.Scale (imgSize),
            transforms.CenterCrop (imgSize),
            transforms.ToTensor (),
            normalize,
        ])
        test_loader = torch.utils.data.DataLoader (
            Myloader (rootpath, test_dir, label_path, test_transform),
            batch_size=int (self.batch_size),
            shuffle=False,
            num_workers=self.n_threads,
            pin_memory=True)
        return train_loader, test_loader


if __name__ == '__main__':
    opt = NetOption()
    data_loader = DataLoader(dataset=opt.data_set, data_path=opt.data_path, label_path=opt.label_path,
                             batch_size=1, rootpath=opt.rootpath,
                             n_threads=opt.nThreads, ten_crop=opt.tenCrop, dataset_ratio=opt.datasetRatio)
    train_loader, test_loader = data_loader.getloader ()
    # weights = [5 if int(k)== 1 else 1 for i, j, k, _ in train_loader]
    # np.save("WsamplerWeight.npy",weights)
    # print(weights)
    for i, j, k, q in test_loader:
        print(i[0].size(2))
        print(k,q)
        break

    # print(len(train_loader))