import os
list1=[]
with open ("/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/3dData.txt", "a") as f:
    path = "/mnt/dataset/splited_Casia2"
    for root, dirs, files in os.walk(path):
        # print(root)
        for i in range(len(files)):
            # print(root+"/"+files[i])
            if files[i].split("_")[0].split("-")[0] !="CS":
                continue
            eyeid = files[i].split("_")[0].split("-")[0]+"-"+files[i].split("_")[0].split("-")[1]
            darkOrLight =  files[i].split("_")[0].split("-")[-1]
            leftOrRight =files[i].split("_")[1]
            list1.append(eyeid+"*"+darkOrLight+"*"+leftOrRight+"\n")
    list1 = set(list1)   
    for i in list1:
        f.writelines(i)
