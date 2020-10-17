"""
从训练验证集中获取eyeid,并根据open narrow_not_synechia narrow分开
"""
import numpy as np

narrownotsynechialist = np.load("/home/yangyifan/code/multiviewCNN_quarter/multiViewCNN/Multi_ViewCNN/dataProcess/add_alone_clock/narrowNotSynechiaList.npy")
synechialist = np.load("/home/yangyifan/code/multiviewCNN_quarter/multiViewCNN/Multi_ViewCNN/dataProcess/add_alone_clock/synechia_exsist_list.npy")
# print(narrownotsynechialist)

def writetxt(lists, path):
    f1 = open(path,"w")
    for i in lists:
        f1.writelines(i+"\n")
    return 0


def main():
    f1 = open("/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess//oneclock_data_split/trainv5.txt")
    f2 = open("/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess//oneclock_data_split/valv5.txt")
    list1 = list(f1.readlines())
    list2 = list(f2.readlines())
    trainlist = []
    vallist = []
    for details in list1:
        trainlist.append(getid(details))

    for details in list2:
        vallist.append(getid(details))

    print(len(set(trainlist)))
    print(len(set(vallist)))
    count1=0
    count2=0
    count3=0
    count=0
    list1 = []
    list2 = []
    list3 = []
    for i in set(trainlist):
        count+=1
        if i in narrownotsynechialist:
            list1.append(i)
            count1+=1
        elif i in synechialist:
            list2.append(i)
            count2+=1
        else:
            list3.append(i)
            count3+=1
    print(">>>>>>>>>>narrow not synechialist num:{}".format(count1))
    for i in list1:
        print(i)
    print(">>>>>>>>>>synechia list num:{}".format(count2))
    for i in list2:
        print(i)
    print(">>>>>>>>>>open list num:{}".format(count3))
    for i in list3:
        print(i)
    # for i in set(vallist):
    #     count+=1
    #     if i in narrownotsynechialist:
    #         count1+=1
    #     elif i in synechialist:
    #         count2+=1
    #     else:
    #         count3+=1
    print(count,count1,count2,count3)



def getid(details):
    split = details.split("_")
    id = split[0]+split[1].capitalize()
    # print(id)
    return id



if __name__ == '__main__':
    main()