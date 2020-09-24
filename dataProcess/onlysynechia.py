import pandas as pd
import numpy as np
####为了representation learning 划分出粘连的钟点

def getlist(path):
    lists = list(open(path).readlines())
    return lists
def getdf1(path):  ####synechia label df
    df = pd.read_csv(path,index_col="Code",encoding="gbk")
    return df
# def getdf2(path):  ####state label df
#     return df
def writetxt(lists,path):
    f1 = open(path,"w")
    for i in lists:
        f1.writelines(i+"\n")
    return 0
def main():
    path = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/oneclock_data_split/trainv5.txt"
    pathdf = "/home/yangyifan/code/multiviewCNN_quarter/multiViewCNN/Multi_ViewCNN/dataProcess/add_alone_clock/one_for_new_clock_labelv1.csv"
    lists = getlist(path)
    syne_df = getdf1(pathdf)
    # state_df = getdf2()
    synelist = []
    for id in lists:
        # print(str(syne_df.loc[id.strip("\n"),"synechia"]))
        if str(syne_df.loc[id.strip("\n"),"synechia"]) == "1":
            synelist.append(id)
    # writepath = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/oneclock_data_split/contrastive_learning/train_representation_159.txt"
    # writetxt(synelist, writepath)

def main1():

    allsynechialist = []
    exsistlist = np.load("/home/yangyifan/code/multiviewCNN_quarter/multiViewCNN/Multi_ViewCNN/dataProcess/add_alone_clock/pathExsistList.npy")
    synechia_used_list = np.load("/home/yangyifan/code/multiviewCNN_quarter/multiViewCNN/Multi_ViewCNN/dataProcess/add_alone_clock/synechia_exsist_list.npy")
    path = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/medical_label/medicaldata.csv"
    df = getdf1(path)
    alllist = list(pd.read_csv(path,encoding="gbk")["Code"])
    for i in range(len(alllist)):
        id = alllist[i]
        # id = eyeid[:-2]
        odoslist = ["Od","Os"]
        for odos in odoslist:
            if df.loc[id,odos+"clock"] != "-" and str(df.loc[id,odos+"clock"]) != "nan":
                # print(id+odos,df.loc[id,odos+"clock"])

                allsynechialist.append(id+odos)
    cannotuselist = set(allsynechialist).difference(set(synechia_used_list))
    print(cannotuselist)
    writetxt(cannotuselist,"synechialist_notexsisit.txt")
    # np.save('synechialist_notexsisit.npy',cannotuselist)
    # generate_command(cannotuselist)

    return
def generate_command(lists):

    index=0
    commandlist = []
    for i in lists:
        index+=1
        id = i[:-2]
        odos = i[-2:]
        if odos =="Od":
            lr = "R"
        elif odos=="Os":
            lr = "L"
        command_d = "find . -name  '"+id+"-"+"D"+"_"+lr+"*'"
        command_l = "find . -name  '" + id + "-" + "L" + "_" + lr + "*'"
        print(command_d)




if __name__ == '__main__':
    main1()
