import numpy as np
import pandas as pd
import random

###for split data
def split_dataset_1(root_path="/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/oneclock_data_split/one_wide_split/wide.txt", tain_percent=0.5, test_percent=0.5, val_percent=0):
    # df = pd.read_csv(root_path)
    all_images1 = open(root_path).readlines()
    # print("all_images", all_images1)
    # all_images = all_images.as_matrix()
    print(len(all_images1))
    all_images=[]
    tmp_list = []
    for img in all_images1:
        if int(img.strip('\n').split("_")[-1])<=5:
            all_images.append(img.strip("\n"))
    print(len(all_images))
    random.shuffle(all_images)
    for img in all_images:
        data = img.split('_')
        idx = data[0]+"_"+data[1]
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
        idx = data[0]+"_"+data[1]
        img +="\n"
        print(img)
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
            # lis=lis+"\n"
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

def testintersection(train_list, test_list):
    print(len(train_list), len(test_list))
    set1 = []
    set2 = []
    for i in train_list:
        data = i.split("_")
        set1.append(data[0] + data[1] + data[2])
    for i in test_list:
        data = i.split("_")
        set2.append(data[0] + data[1] + data[2])
    print(set(set1))
    print(len(set(set1)), len(set(set2)))
    print(set(set1).intersection(set(set2)))

def shuffleit():
    trainlist = open("/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/oneclock_data_split/one_openandnarrow_split/train.txt").readlines()
    vallist = open("/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/oneclock_data_split/one_openandnarrow_split/val.txt").readlines()
    random.shuffle(trainlist)
    random.shuffle(vallist)
    write_to_txt(trainlist, "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/oneclock_data_split/one_openandnarrow_split/train_all.txt")
    write_to_txt(vallist, "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/oneclock_data_split/one_openandnarrow_split/val_all.txt" )

if __name__ == '__main__':
    # shuffleit()
    # train_list, test_list, val_list = split_dataset_1()
    # write_to_txt(train_list,"/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/oneclock_data_split/one_wide_split/train_wide.txt")
    # write_to_txt(test_list,
    #              "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/oneclock_data_split/one_wide_split/val_wide.txt")
    # with open("/home/yangyifan/code/multiViewCNN/multi-viewCNN/dataProcess/3dData.txt") as f:
    #     all = sorted(f.readlines())  ##all file structure in file
    #     for i in all:
    #         new_i = i.split("*")[0]+"_"+i.split("*")[1]+"_"+i.split("*")[2]
    #         print(new_i)
    # val_list, test_list, train_list = split_dataset_1 ()
    # val_list = train_list
    # final_val_list=[]
    # # print(val_list)
    # for i in val_list:
    #     print(i)
    #     newid_dark = i.split("_")[0]+"*D"
    #     newid_light = i.split ("_")[0] + "*L"
    #     if i.split("_")[1] =="os":
    #         newid_dark+="*L\n"
    #         newid_light+="*L\n"
    #     elif i.split("_")[1] =="od":
    #         newid_dark += "*R\n"
    #         newid_light += "*R\n"
    #     print(">>>>>>>>>>>>>>>>",newid_dark)
    #     if newid_dark in all and newid_light in all:
    #         final_val_list.append(i)
    # print(len(val_list),len(final_val_list))
    # write_to_txt(final_val_list, "/home/yangyifan/code/multiViewCNN/multi-viewCNN/dataProcess/train.txt")
    # # spilted_eyeid_to_txt (train_list)
    # # spilted_eyeid_to_txt (train_list)

    list1 = open("/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/wide_split/quartersplit/train_quater1.txt").readlines()
    list2 =open("/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/wide_split/quartersplit/val_quater1.txt").readlines()
    testintersection(list1, list2)
