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
import glob

def pil_loader(path):
    return Image.open(path).convert("RGB")


def get_label(label_dir):
    label_df = pd.read_csv(label_dir)
    label_df = label_df.set_index('details')
    return label_df

def getone(realpath, indexs, region, flag):
    res = {}
    res["imagelist_half1"]=[]
    res["imagelist_half2"]=[]
    res["region_half1"]=""
    res["region_half2"]=""
    res["vertical_half1"]=-1
    res["vertical_half2"]=-1
    if flag == -1000:
        pathlist = [(realpath + str(indexs), region, indexs)]
        for path, region, i in pathlist:
            vertical_str = "/vertical_" + region[0] + ".npy"
            vertical = int(np.load(path + vertical_str))
            all_image_path = glob.glob(os.path.join(path, "*.jpg"))
            all_image_path = sorted(all_image_path)
            res["imagelist_half1"] = all_image_path[:21]
            res["vertical_half1"] = vertical
            res["region_half1"] = region
    #########################overlap method#####################################
    else :
        ####21 images
        if int(flag)==0:
            step = int(flag) * 1
            top = 21 - step
            buttom = 25
        elif int(flag)>0:
            step = int(flag)*1
            top = 21-step
            buttom = -step
        elif int(flag)<0:
            step = int(flag)*1
            top = -step
            buttom = -21-step
        # print("overlap loaddata method")
        if region == "right" and int(flag)<0:
            indexpath = (int(indexs)+1)%6
            if indexpath>int(indexs):
                pathlist = [(realpath + str(indexs), 'right', int(indexs)), (realpath + str(indexpath), 'right', indexpath)]
            else:
                pathlist=[(realpath + str(indexs), 'right', int(indexs)), (realpath + str(indexpath), 'left', indexpath)]
            for path, region, i in pathlist:
                vertical_str = "/vertical_" + region[0] + ".npy"
                vertical = int(np.load(path + vertical_str))
                all_image_path = glob.glob(os.path.join(path, "*.jpg"))
                all_image_path = sorted(all_image_path)
                if i == int(indexs):
                    res["imagelist_half1"] = all_image_path[buttom:]
                    res["vertical_half1"] = vertical
                    res["region_half1"] = region
                if i == indexpath:
                    res["imagelist_half2"] = all_image_path[:top]
                    res["vertical_half2"] = vertical
                    res["region_half2"] = region
        elif region == "left" and int(flag)<0:
            indexpath = (int(indexs) + 1) % 6
            if indexpath > int(indexs):
                pathlist = [(realpath + str(indexs), 'left', int(indexs)),(realpath + str(indexpath), 'left', indexpath)]
            else:
                pathlist = [(realpath + str(indexs), 'left', int(indexs)),(realpath + str(indexpath), 'right', indexpath)]

            for path, region, i in pathlist:
                vertical_str = "/vertical_" + region[0] + ".npy"
                vertical = int(np.load(path + vertical_str))
                all_image_path = glob.glob(os.path.join(path, "*.jpg"))
                all_image_path = sorted(all_image_path)
                if i == int(indexs):
                    res["imagelist_half1"] = all_image_path[buttom:]
                    res["vertical_half1"] = vertical
                    res["region_half1"] = region
                if i == indexpath:
                    res["imagelist_half2"] = all_image_path[:top]
                    res["vertical_half2"] = vertical
                    res["region_half2"] = region
        if region == "right" and int(flag) >= 0:
            indexpath = (int(indexs) - 1) % 6
            if indexpath < int(indexs):
                pathlist = [(realpath + str(indexs), 'right', int(indexs)),
                            (realpath + str(indexpath), 'right', indexpath)]
            else:
                pathlist = [(realpath + str(indexs), 'right', int(indexs)),
                            (realpath + str(indexpath), 'left', indexpath)]
            for path, region, i in pathlist:
                vertical_str = "/vertical_" + region[0] + ".npy"
                vertical = int(np.load(path + vertical_str))
                all_image_path = glob.glob(os.path.join(path, "*.jpg"))
                all_image_path = sorted(all_image_path)
                if i == int(indexs):
                    res["imagelist_half2"] = all_image_path[:top]
                    res["vertical_half2"] = vertical
                    res["region_half2"] = region
                if i == indexpath:
                    res["imagelist_half1"] = all_image_path[buttom:]
                    res["vertical_half1"] = vertical
                    res["region_half1"] = region
        elif region == "left" and int(flag) >= 0:
            indexpath = (int(indexs) - 1) % 6
            if indexpath < int(indexs):
                pathlist = [(realpath + str(indexs), 'left', int(indexs)),
                            (realpath + str(indexpath), 'left', indexpath)]
            else:
                pathlist = [(realpath + str(indexs), 'left', int(indexs)),
                            (realpath + str(indexpath), 'right', indexpath)]

            for path, region, i in pathlist:
                vertical_str = "/vertical_" + region[0] + ".npy"
                vertical = int(np.load(path + vertical_str))
                all_image_path = glob.glob(os.path.join(path, "*.jpg"))
                all_image_path = sorted(all_image_path)
                if i == int(indexs):
                    res["imagelist_half2"] = all_image_path[:top]
                    res["vertical_half2"] = vertical
                    res["region_half2"] = region
                if i == indexpath:
                    res["imagelist_half1"] = all_image_path[buttom:]
                    res["vertical_half1"] = vertical
                    res["region_half1"] = region

                    # res.append((all_image_path[:11], vertical, path, region))
    print("flag:", flag, realpath, indexs, region)
    print(res)
    if flag!=-1:
        if len(res["imagelist_half1"])+len(res["imagelist_half2"])!=16:

            print("!!!!>>>!!!",len(res["imagelist_half1"])+len(res["imagelist_half2"]))
    # a = all_image_path
    return res

def make_dataset(rootpath, root, label_df):
    images_light = []
    images_dark = []
    for line in open (root):
        org_path = line.strip("\n")
        splits = org_path.split ("_")
        try:
            flag = splits[4]
        except:
            flag = -1000
        if flag ==-1000:
            label = label_df.loc[org_path, "synechia"]
        else:
            splits=org_path.split("_")
            key = splits[0]+"_"+splits[1]+"_"+splits[2]+"_"+splits[3]
            label = label_df.loc[key , "synechia"]
            # print("orgpath",org_path,"label:",label)
        eyeid = splits [0]
        odos = splits[1]
        region = splits[2]
        indexs = splits[3]
        realpath = rootpath + "/" + eyeid + "/"
        if odos == "od":
            realpath += "R"
        elif odos == "os":
            realpath += "L"
        # print(realpath)
        darkrealpath = realpath + "/D/"
        lightrealpath = realpath + "/L/"
        res_d = getone(darkrealpath, int(indexs), region, flag)
        res_l = getone(lightrealpath, int(indexs), region, flag)
        res_l["label"] = label
        res_l["details"] = org_path
        res_d["label"] = label
        res_d["details"] = org_path
        images_light.append(res_l)
        images_dark.append(res_d)

    return images_light, images_dark

def make3d(dicts, transform):   ##dict   {"imagelist_half1":imagelist1, "vertical_half1":vertical_half1, "region_half1":region_half1,,,half2,,,,"lalel":label,"details":CS-001_od_left_1}
    try:
        imgs = Image.open(dicts["imagelist_half1"][0]).convert("RGB")
    except:
        imgs = Image.open(dicts["imagelist_half2"][0]).convert("RGB")
    lists = [dicts["imagelist_half1"], dicts["imagelist_half2"]]
    label = dicts["label"]
    regionlist = [dicts["region_half1"], dicts["region_half2"]]
    regionCor_l = (0, 0, imgs.size[0] / 2, imgs.size[1])
    regionCor_r = (imgs.size[0] / 2, 0, imgs.size[0], imgs.size[1])
    vertical_centerlist = [dicts["vertical_half1"], dicts["vertical_half2"]]
    vrc = VerticalCrop(244)
    rgc = RandomGammaCorrection()
    rgc.randomize_parameters()
    imglist = []
    fullimglist = []
    index=0
    details = dicts["details"]
    for imgpathlist in lists:
        if imgpathlist==[]:
            continue
        for imgpath in imgpathlist:
            region = regionlist[index]
            if region=="right":
                regionCor = regionCor_r
            elif region=="left":
                regionCor = regionCor_l
            fullimg = Image.open(imgpath).convert("RGB")
            orgimage = fullimg.crop(regionCor)
            vertical_center = vertical_centerlist[index]
            vrc.randomize_parameters(vertical_center)
            crop_image = np.asarray(vrc(orgimage))
            crop_image = np.asarray(rgc(crop_image))
            img = Image.fromarray(crop_image, 'RGB')
            if region == "right":
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                # img.save('/home/yangyifan/save/crop_RU.jpg')
            imglist.append(transform(img))
            fullimglist.append(transform(fullimg))
        # print(index)
        index += 1

    input = torch.stack (imglist).permute(1, 0, 2, 3)
    fullinput = torch.stack (fullimglist).permute(1, 0, 2, 3)
    return input, label, fullinput, details

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
        dark_input, label, dark_full_input, details = make3d(images_dark,   self.transform)
        light_input, label, light_full_input, details = make3d (images_light,   self.transform)
        return (dark_input, dark_full_input), (light_input, light_full_input), label, details

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