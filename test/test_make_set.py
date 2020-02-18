import os
# line = "CS-015_od_left_0"
root1 = "/home/yangyifan/code/multiViewCNN/multi-viewCNN/dataProcess/for_data_split/new_data_split/train.txt"
root2 = "/home/yangyifan/code/multiViewCNN/multi-viewCNN/dataProcess/for_data_split/new_data_split/val.txt"
def make_dataset(rootpath, root, label_df):
    images_light = []
    images_dark = []
    for line in open (root):
        org_path = line.strip ('\n')
        label = label_df.loc[org_path, "synechia"]
        eyeid = org_path.split ("_")[0]
        odos = org_path.split ("_")[1]
        region = org_path.split ("_")[2]
        indexs = int(org_path.split ("_")[3])
        realpath = rootpath + "/" + eyeid + "/"
        if odos == "od":
            realpath += "R"
        elif odos == "os":
            realpath += "L"
        darkrealpath = realpath + "/D/"
        lightrealpath = realpath + "/L/"
        lightrealpath += str (int(indexs/2))
        darkrealpath += str (int(indexs/2))
        all_image_path = list (os.listdir (lightrealpath))
        all_image_path.sort ()
        if indexs/2 - int(indexs/2) ==0.5:
            images_light.append ((all_image_path[11:21], lightrealpath, region))
            all_image_path1 = list (os.listdir (darkrealpath))
            all_image_path1.sort ()
            images_dark.append ((all_image_path1[11:21], darkrealpath, region))
        if indexs/2 - int(indexs/2) ==0:
            images_light.append ((all_image_path[1:11], lightrealpath, region))
            all_image_path1 = list (os.listdir (darkrealpath))
            all_image_path1.sort ()
            images_dark.append ((all_image_path1[1:11], darkrealpath, region))
    # print(images_dark)
    return images_light, images_dark

if __name__ == '__main__':
    root_path = "/mnt/dataset/splited_Casia2"
    make_dataset(root_path,root2,"")