import numpy as np
import pandas as pd
import random
def split_dataset_1(root_path="/home/yangyifan/code/multiViewCNN/multi-viewCNN/dataProcess/3dlabel_only_narrow.csv", tain_percent=0.34, test_percent=0.34, val_percent=0.3):
    df = pd.read_csv(root_path)
    all_images = df['details']

    all_images = all_images.as_matrix()
    random.shuffle(all_images)
    tmp_list = []

    for img in all_images:
        data = img.split('_')
        idx = data[0]
        if idx not in tmp_list:
            tmp_list.append(idx)

    random.shuffle(tmp_list)
    train_num = int(len(tmp_list) * tain_percent)
    test_num = int(len(tmp_list) * test_percent)
    train_tmp_list = tmp_list[:train_num]
    val_tmp_list = tmp_list[train_num+test_num:]
    train_list = []
    test_list = []
    val_list = []
    for img in all_images:
        data = img.split('_')
        idx = data[0]
        if idx in val_tmp_list:
            val_list.append(img)
        elif idx in train_tmp_list:
            train_list.append(img)
        else:
            test_list.append(img)
    return train_list, test_list, val_list


def write_to_txt(list,pathtxt):
    with open(pathtxt, "w") as f:
        for lis in list:
            print (lis)
            lis=lis+"\n"
            f.write(lis)

def spilted_eyeid_to_txt(lists):
    dark_list = []
    light_list = []
    for lis in lists:
        newlis_org = lis.split("_")[0]
        newlis1= newlis_org+"*D"
        newlis2 = newlis_org+"*L"
        if lis.split("_")[1]=="od":
            newlis1 += "*R"
            newlis2 += "*R"
        if lis.split ("_")[1] == "os":
            newlis1 += "*L"
            newlis2 += "*L"
        if newlis1+"\n" in result1 and newlis1 not in dark_list:
            # print("innn")
            dark_list.append(newlis1)
        if newlis2+"\n" in result2 and newlis2 not in light_list:
            # print("innn")
            light_list.append(newlis2)
    print (dark_list)
    print (light_list)
    # write_to_txt (dark_list, "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/train_dark.txt")
    # write_to_txt (light_list, "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/train_light.txt")

if __name__ == '__main__':

    with open("/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/3dData.txt") as f:
        eyelist = []
        result1 = []
        result2 = []
        result3 = []
        result4 = []
        all = sorted(f.readlines())  ##all file structure in file
        for line in all:
            eyeid = line.split("*")[0]
            if eyeid not in eyelist:
                eyelist.append(eyeid)
        for i in eyelist:
            id1 = i + "*" +"D"+"*"+"L\n"     ##dark left
            id2 = i + "*" +"L"+"*"+"L\n"     ##light_left
            id3 = i + "*" +"D"+"*"+"R\n"     ##dark_right
            id4 = i + "*" + "L" + "*" + "R\n"##light_right
            if id1 in all and id2 in all:
                result1.append(id1)           ##result1 dark
                result2.append(id2)           ##result2 light
            if id3 in all and id4 in all:
                result1.append(id3)
                result2.append(id4)
        # for i in result1:
        #     result3.append (i.strip("\n") + "*0")    ##result3 dark 0-5
        #     result3.append (i.strip("\n") + "*1")
        #     result3.append (i.strip("\n") + "*2")
        #     result3.append (i.strip("\n") + "*3")
        #     result3.append (i.strip("\n") + "*4")
        #     result3.append (i.strip("\n") + "*5")
        # for i in result2:
        #     result4.append (i.strip("\n") + "*0")   ####result4 light 0-5
        #     result4.append (i.strip("\n") + "*1")
        #     result4.append (i.strip("\n") + "*2")
        #     result4.append (i.strip("\n") + "*3")
        #     result4.append (i.strip("\n") + "*4")
        #     result4.append (i.strip("\n") + "*5")

    # dark_list = []
    # light_list = []
    # val_list, test_list, train_list = split_dataset_1 ()
    # for lis in val_list:
    #     newlis_org = lis.split("_")[0]
    #     newlis1= newlis_org+"*D"
    #     newlis2 = newlis_org+"*L"
    #     if lis.split("_")[1]=="od":
    #         newlis1 += "*R"
    #         newlis2 += "*R"
    #     if lis.split ("_")[1] == "os":
    #         newlis1 += "*L"
    #         newlis2 += "*L"
    #     if newlis1+"\n" in result1 and newlis1 not in dark_list:
    #         print("innn")
    #         dark_list.append(newlis1)
    #     if newlis2+"\n" in result2 and newlis2 not in light_list:
    #         print("innn")
    #         light_list.append(newlis2)
    # print(dark_list)
    # print(light_list)
    # # write_to_txt (dark_list, "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/train_dark.txt")
    # # write_to_txt (light_list, "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/train_light.txt")
    val_list, test_list, train_list = split_dataset_1 ()
    spilted_eyeid_to_txt(test_list)
    spilted_eyeid_to_txt (train_list)
    spilted_eyeid_to_txt (train_list)