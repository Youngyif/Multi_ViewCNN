import pandas as pd
import os


def get_label(label_dir):
    label_df = pd.read_csv(label_dir)
    label_df = label_df.set_index('details')
    return label_df

newtraintxtpath = "/home/yangyifan/code/multiViewCNN/multi-viewCNN/dataProcess/for_data_split/new_data_split/train2d.txt"
newvaltxtpath = "/home/yangyifan/code/multiViewCNN/multi-viewCNN/dataProcess/for_data_split/new_data_split/val2d.txt"
rootpath = "/mnt/dataset/splited_Casia2"
path = "/home/yangyifan/code/multiViewCNN/multi-viewCNN/dataProcess/for_data_split/new_data_split/train.txt" ###path for train.txt/val.txt
path1 = "/home/yangyifan/code/multiViewCNN/multi-viewCNN/dataProcess/for_data_split/new_data_split/val.txt"
label_dir = "/home/yangyifan/code/multiViewCNN/multi-viewCNN/dataProcess/label_version/v2_exisit_noaloneclock_half_3d_label.csv"
label_df = get_label(label_dir)
images_light=[]
images_dark=[]

def transferit(path):
    with open(path) as f:
        for line in  f.readlines():
            org_path = line.strip('\n')
            label = label_df.loc[org_path, "synechia"]
            eyeid = org_path.split("_")[0]
            odos = org_path.split("_")[1]
            region = org_path.split("_")[2]
            indexs = int(org_path.split("_")[3])
            realpath = rootpath + "/" + eyeid + "/"
            if odos == "od":
                realpath += "R"
            elif odos == "os":
                realpath += "L"
            darkrealpath = realpath + "/D/"
            lightrealpath = realpath + "/L/"
            lightrealpath += str(int(indexs / 2))
            darkrealpath += str(int(indexs / 2))
            all_image_path = list(os.listdir(lightrealpath))
            all_image_path.sort()
            if indexs / 2 - int(indexs / 2) == 0.5:
                images_light.append((all_image_path[11:21], lightrealpath, label, region))
                all_image_path1 = list(os.listdir(darkrealpath))
                all_image_path1.sort()
                images_dark.append((all_image_path1[11:21], darkrealpath, label, region))
            if indexs / 2 - int(indexs / 2) == 0:
                images_light.append((all_image_path[1:11], lightrealpath, label, region))
                all_image_path1 = list(os.listdir(darkrealpath))
                all_image_path1.sort()
                images_dark.append((all_image_path1[1:11], darkrealpath, label, region))
    return images_light,images_dark

if __name__ == '__main__':
    images_light, images_dark = transferit(path1)
    sum = 0
    res=[]
    res1=[]
    with open(newvaltxtpath,"a") as f1:
        for i,j,k,l in images_light:
            for q in i:
                sum+=1
                res.append(j+"/"+q)
        for i, j, k, l in images_dark:
            for q in i:
                sum += 1
                res1.append(j + "/" + q+"#"+str(k)+"#"+l)
        for i,j in zip(res,res1):
            # print(i+"#"+j+"\n")
            f1.writelines(i+"#"+j+"\n")
    # with open(newvaltxtpath,"a") as f2:
    #     for i,j,k,l in images_:
    #         for q in i:
    #             sum+=1
    #             res = j+"/"+q
    #             f2.writelines(res)


