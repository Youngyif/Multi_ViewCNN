import os
from PIL import Image
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
import torchvision.models as models
import pandas as pd
# eyeid = "CS-002"
# flag = "R"
# root = "/mnt/dataset/splited_Casia2/"
# path = root + eyeid + "/" + flag

# def extractclock(path):
#     all_clock = []
#     for i in range (6):
#         pathv1 = path + "/" + str (i)
#         print(os.path.exists (pathv1))
#         all_image_path = list (os.listdir (pathv1))
#         all_image_path.sort ()
#         # for filename in all_image_path[0:21]:
#         #     print(filename)
#         all_clock.append ((pathv1, all_image_path[0:21]))
#     return all_clock

# def extractFile(net):         ##获取21张图片的路径
#     # print(os.path.exists(path_L))
#     if os.path.exists (path):
#         path_d = path + "/D"
#         darklist = extractclock (path_d)
#         model_demo (darklist, net)
#         path_l = path + "/L"
#         lightlist = extractclock (path_l)

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
        all_image_path = list (os.listdir (lightrealpath))
        all_image_path.sort ()
        images_light.append ((all_image_path[0:21], lightrealpath, label, region))
        all_image_path1 = list (os.listdir (darkrealpath))
        all_image_path1.sort ()
        images_dark.append ((all_image_path1[0:21], darkrealpath, label, region))

    return images_light, images_dark

def make3d(tup):   #tup:([21 images], label, region)
    imgs = Image.open (tup[1] + "/" + tup[0][0]).convert ("RGB")
    normalize = transforms.Normalize (mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    transform = transforms.Compose ([
        # transforms.Scale(600),
        transforms.Scale (520),
        transforms.CenterCrop (512),
        transforms.ToTensor (),
        normalize,
    ])
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
        dark_input, label = make3d(images_dark)
        light_input, label = make3d (images_light)
        return dark_input, light_input, label

    def __len__(self):
        return len (self.images_light)
