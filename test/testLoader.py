import os
import pandas as pd
import os
from PIL import Image
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
import torchvision.models as models
import pandas as pd

def get_label(label_dir):
    label_df = pd.read_csv(label_dir)
    label_df = label_df.set_index('details')
    return label_df

def make_dataset(rootpath, root, label_df):
    images_light = []
    images_dark = []
    label_col_name=""
    for line in open (root):
        org_path = line.rstrip ('\n')
        label = label_df.loc[org_path, "synechia"]
        eyeid = org_path.split("_")[0]
        odos = org_path.split("_")[1]
        region = org_path.split("_")[2]
        indexs = org_path.split("_")[3]
        # print(org_path)
        realpath = rootpath+"/"+eyeid+"/"
        if odos == "od":
            realpath+="R"
        elif odos =="os":
            realpath+="L"
        darkrealpath = realpath+"/D/"
        lightrealpath = realpath+"/L/"
        lightrealpath+=str(indexs)
        darkrealpath+=str(indexs)
        all_image_path = list (os.listdir (lightrealpath))
        all_image_path.sort ()
        images_light.append ((all_image_path[0:21], lightrealpath, label, region))
        all_image_path = list (os.listdir (darkrealpath))
        all_image_path.sort ()
        images_dark.append ((all_image_path[0:21], darkrealpath, label, region))
 
    return (images_light, images_dark)

def make3d(rootpath, tup):   #tup:([21 images], label, region)
    print(tup[1]+"/"+tup[0][0])
    imgs = Image.open(tup[1]+"/"+tup[0][0]).convert("RGB")
    normalize = transforms.Normalize (mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    transform = transforms.Compose ([
        # transforms.Scale(600),
        transforms.Scale (520),
        transforms.CenterCrop (512),
        transforms.ToTensor (),
        normalize,
    ])
    lists =tup[0]
    rootpath = tup[1]
    label=tup[2]
    region=tup[3]
    if region =="left":
        regionCor = (0, 0, imgs.size[0] / 2, imgs.size[1])
    elif region =="right":
        regionCor = (imgs.size[0]/2, 0, imgs.size[0], imgs.size[1])
    print(">>>>>>>>>", regionCor)
    imgs = [transform (Image.open (rootpath+"/" + imgpath).convert ("RGB").crop(regionCor)) for imgpath in lists]
    input = torch.stack (imgs)
    print(input.shape)
    return input, label

if __name__ == '__main__':
    root = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/train.txt"
    rootpath = "/mnt/dataset/splited_Casia2"
    label_dir = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/3dlabel_only_narrow.csv"
    df = get_label (label_dir)
    images_light, images_dark = make_dataset(rootpath, root, df)
    input, label= make3d (rootpath, images_light[0])