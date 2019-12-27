import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
import torchvision.models as models

eyeid = "CS-002"
flag = "R"
root = "/mnt/dataset/splited_Casia2/"
path = root + eyeid + "/" + flag

def extractclock(path):
    all_clock = []
    for i in range (6):
        pathv1 = path + "/" + str (i)
        print(os.path.exists (pathv1))
        all_image_path = list (os.listdir (pathv1))
        all_image_path.sort ()
        # for filename in all_image_path[0:21]:
        #     print(filename)
        all_clock.append ((pathv1, all_image_path[0:21]))
    return all_clock

def extractFile(net):         ##获取21张图片的路径
    # print(os.path.exists(path_L))
    if os.path.exists (path):
        path_d = path + "/D"
        darklist = extractclock (path_d)
        model_demo (darklist, net)
        path_l = path + "/L"
        lightlist = extractclock (path_l)

class Myloader(data.Dataset):
    def __init__(self, root_dark, root_light, label_dir, transform=None):
        self.root_dark = root_dark
        self.root_light = root_light
        self.label_dir = label_dir
        self.loader = pil_loader
        self.transform = transform

        self.label_df = get_label(label_dir)
        self.imgs = make_dataset(root_dark, root_light, self.label_df)


    def __getitem__(self, index):
        path_dark, path_light, path_dark_mask, path_light_mask, left, right = self.imgs[index]
        ##
        img_dark = self.loader(path_dark)
        dark_mask = Image.open(path_dark_mask).convert("L")
        left_region = (0, 0, img_dark.size[0]/2, img_dark.size[1])
        right_region = (img_dark.size[0]/2, 0, img_dark.size[0], img_dark.size[1])
        img_dark_leftRegion = img_dark.crop (left_region)
        img_dark_mask_leftRegion = img_dark_mask.crop(left_region)
        # concate_dark_leftRegion = np.concatenate(img_dark_mask, img_dark_mask_leftRegion)

        img_dark_rightRegion = img_dark.crop (right_region)
        img_dark_mask_rightRegion = img_dark_mask.crop(right_region)
        # print("size after crop", np.max(np.array(img_dark_rightRegion)))
        img_dark_leftRegion= self.transform (img_dark_leftRegion)
        img_dark_rightRegion = self.transform (img_dark_rightRegion)
        ##
        img_light = self.loader(path_light)

        img_light_leftRegion = img_light.crop (left_region)
        img_light_rightRegion = img_light.crop (right_region)
        img_light_leftRegion= self.transform (img_light_leftRegion)
        img_light_rightRegion = self.transform (img_light_rightRegion)

        img = (img_dark_leftRegion, img_dark_rightRegion, img_light_leftRegion, img_light_rightRegion)
        labels = (left, right)
        return img, labels

    def __len__(self):
        return len (self.imgs)
