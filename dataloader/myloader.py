import os
from PIL import Image
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
import torchvision.models as models
import pandas as pd

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
        """
        尝试一下分类宽窄角
        """
        labelopennarrow = label_df.loc[org_path, "openORnarrow"]
        label = labelopennarrow
        #####
        eyeid = org_path.split ("_")[0]
        odos = org_path.split ("_")[1]
        region = org_path.split ("_")[2]
        indexs = int (org_path.split ("_")[3])
        realpath = rootpath + "/" + eyeid + "/"
        if odos == "od":
            realpath += "R"
        elif odos == "os":
            realpath += "L"
        darkrealpath = realpath + "/D/"
        lightrealpath = realpath + "/L/"
        lightrealpath += str (int (indexs / 2))
        darkrealpath += str (int (indexs / 2))
        all_image_path = list (os.listdir (lightrealpath))
        all_image_path.sort ()
        if indexs / 2 - int (indexs / 2) == 0.5:
            images_light.append ((all_image_path[11:21], lightrealpath, label, region))
            all_image_path1 = list (os.listdir (darkrealpath))
            all_image_path1.sort ()
            images_dark.append ((all_image_path1[11:21], darkrealpath, label, region))
        if indexs / 2 - int (indexs / 2) == 0:
            images_light.append ((all_image_path[1:11], lightrealpath, label, region))
            all_image_path1 = list (os.listdir (darkrealpath))
            all_image_path1.sort ()
            images_dark.append ((all_image_path1[1:11], darkrealpath, label, region))
    # print(images_dark)
    return images_light, images_dark

def make3d(tup, transform):   #tup:([10 images], label, region)
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
    imgs = [transform (Image.open (rootpath + "/" + imgpath).convert ("RGB").crop (regionCor)) for imgpath in lists]
    input = torch.stack (imgs)
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
        dark_input, label = make3d(images_dark,  self.transform)
        light_input, label = make3d (images_light, self.transform)
        return dark_input, light_input, label,images_dark[0]

    def __len__(self):
        return len (self.images_light)
