import os
from PIL import Image
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import numpy as np
import random

def pil_loader(path):
    return Image.open(path).convert("RGB")


def get_label(label_dir):
    label_df = pd.read_csv(label_dir)
    label_df = label_df.set_index('details')
    return label_df


def make_dataset(rootpath, root, label_df):
    images_light = []
    images_dark = []
    for line in open (root):
        org_path = line.strip ('\n')
        label = label_df.loc[org_path, "synechia"]
        eyeid = org_path.split ("_")[0]
        odos = org_path.split ("_")[1]
        region = org_path.split ("_")[2]
        indexs = org_path.split ("_")[3]
        realpath = rootpath + "/" + eyeid + "/"
        if odos == "od":
            realpath += "R"
        elif odos == "os":
            realpath += "L"
        darkrealpath = realpath + "/D/"
        lightrealpath = realpath + "/L/"
        lightrealpath += str (indexs)
        darkrealpath += str (indexs)
        if region =="left":
            vertical_light = int(np.load(lightrealpath + "/vertical_l.npy"))
            vertical_dark = int(np.load(darkrealpath + "/vertical_l.npy"))
        if region =="right":
            vertical_light = int(np.load(lightrealpath + "/vertical_r.npy"))
            vertical_dark = int(np.load(darkrealpath + "/vertical_r.npy"))
        all_image_path = list (os.listdir (lightrealpath))
        all_image_path.sort ()
        images_light.append ((all_image_path[0:21], lightrealpath, label, region, vertical_light))
        all_image_path1 = list (os.listdir (darkrealpath))
        all_image_path1.sort ()
        images_dark.append ((all_image_path1[0:21], darkrealpath, label, region, vertical_dark))

    return images_light, images_dark

def make3d(tup, transform):   #tup:([21 images], label, region)
    imgs = Image.open (tup[1] + "/" + tup[0][0]).convert ("RGB")
    # normalize = transforms.Normalize (mean=[0.485, 0.456, 0.406],
    #                                   std=[0.229, 0.224, 0.225])
    #
    # transform = transforms.Compose ([
    #     transforms.Scale (520),
    #     transforms.CenterCrop (512),
    #     transforms.ToTensor (),
    #     normalize,
    # ])
    lists = tup[0]
    rootpath = tup[1]
    label = tup[2]
    region = tup[3]
    if region == "left":
        regionCor = (0, 0, imgs.size[0] / 2, imgs.size[1])
    elif region == "right":
        regionCor = (imgs.size[0] / 2, 0, imgs.size[0], imgs.size[1])
    vertical_center = tup[4]
    vrc = VerticalCrop(244)
    rgc = RandomGammaCorrection()
    rgc.randomize_parameters()
    vrc.randomize_parameters(vertical_center)
    imglist = []
    for imgpath in lists:
        orgimage = Image.open(rootpath + "/" + imgpath).convert("RGB").crop(regionCor)
        crop_image = np.asarray(vrc(orgimage))
        crop_image = np.asarray(rgc(crop_image))
        img = Image.fromarray(crop_image, 'RGB')
        if region == "right":
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img.save('/home/yangyifan/save/crop.jpg')
        imglist.append(transform(img))
    # imgs = [transform (Image.open (rootpath + "/" + imgpath).convert ("RGB").crop (regionCor)) for imgpath in lists]
    input = torch.stack (imglist)
    return input, label

class Myloader(data.Dataset):
    def __init__(self, rootpath, txtroot, label_dir, transform=None):  ##root: path before filename
        self.root = txtroot
        self.label_dir = label_dir
        self.loader = pil_loader
        self.transform = transform
        self.label_df = get_label(label_dir)
        self.rootpath = rootpath
        self.images_light, self.images_dark = make_dataset(rootpath, txtroot, self.label_df)


    def __getitem__(self, index):
        images_light = self.images_light[index]
        images_dark = self.images_dark[index]
        dark_input, label = make3d(images_dark,   self.transform)
        light_input, label = make3d (images_light,   self.transform)
        return dark_input, light_input, label, images_dark[0]

    def __len__(self):
        return len (self.images_light)



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