import pandas as pd

def splitdata():
    lists = list(open("/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/oneclock_data_split/one_openandnarrow_split/train_all.txt").readlines())
    print(lists)
    finallist = []
    for i in lists:
        index = i.strip("\n").split("_")[-1]
        if index in ["0","1"]:
            finallist.append(i)
    print(finallist,len(finallist))
    f = open("/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/wide_split/quartersplit/train_quater1.txt", "a")
    for i in finallist:
        f.writelines(i)

    return
label_dir = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/wide_split/label for all/label_half_opennarrow.csv"
def changelabels():
    label_df = pd.read_csv(label_dir)
    label_df = label_df.set_index('details')
    label_df.info()
    lists1 = list(open("/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/wide_split/quartersplit/val_quater.txt"))
    lists2 = list(open("/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/wide_split/quartersplit/train_quater.txt"))
    lists1.extend(lists2)
    # print(len(lists1))
    labels =[]
    for i in lists1:
        id = i.split("_")
        orgid = id[0]+"_"+id[1]+"_"+id[2]+"_"
        index = int(id[-1])
        max = 0
        ids = [orgid+str(0+index*6),orgid+str(1+index*6),orgid+str(2+index*6), orgid+str(3+index*6), orgid+str(4+index*6), orgid+str(5+index*6)]
        for j in ids:
            a = label_df.loc[j,"synechia"]
            if int(a)>max:
                max = a
        labels.append(max)
        # print(i,max)
    dict = {"details":[i.strip("\n") for i in lists1], "synechia":labels}
    df = pd.DataFrame(dict)
    df.to_csv("/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/wide_split/label for all/label_quarter.csv")


###############to volume#############
def splitdata1():
    lists = list(open("/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/wide_split/train_all.txt").readlines())
    print(lists)
    finallist = []
    for i in lists:
        index = i.strip("\n").split("_")[-1]
        if index == "0":
            finallist.append(i)
    print(finallist,len(finallist))
    f = open("/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/wide_split/volumesplit/train_volume.txt", "a")
    for i in finallist:
        f.writelines(i)

    return
label_dir = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/wide_split/label for all/label_half_opennarrow.csv"
def changelabels1():
    label_df = pd.read_csv(label_dir)
    label_df = label_df.set_index('details')
    label_df.info()
    lists1 = list(open("/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/wide_split/volumesplit/train_volume.txt"))
    lists2 = list(open("/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/wide_split/volumesplit/val_volume.txt"))
    lists1.extend(lists2)
    # print(len(lists1))
    labels =[]
    for i in lists1:
        id = i.split("_")
        orgid = id[0]+"_"+id[1]+"_"+id[2]+"_"
        max = 0
        ids = [orgid+str(i) for i in range(12)]
        for j in ids:
            a = label_df.loc[j,"synechia"]
            if int(a)>max:
                max = a
        labels.append(max)
        # print(i,max)
    dict = {"details":[i.strip("\n") for i in lists1], "synechia":labels}
    df = pd.DataFrame(dict)
    df.to_csv("/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/wide_split/volumesplit//label_volume.csv")



if __name__ == '__main__':
    splitdata()
    # changelabels1()