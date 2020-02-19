import pandas as pd
import numpy as np

def creatCsv(eyeId_list): ##genetate a csv with all label equal to 0
    details = []
    eyeid = []
    odOros = []
    indexs = []
    CorpedRegion = []
    synechia=[]
    state=[]
    for id in eyeId_list:
        od_os = id.split("_")[-1]
        ids = id.split("_")[0]
        # for index in range(12):
        #     eyeid.append(id)
        #     details.append(id+"_od"+"_left_"+str(index))
        #     odOros.append("od")
        #     indexs.append(index)
        #     CorpedRegion.append("left")
        #     synechia.append(0)
        #     state.append(0)
        # for index in range (12):
        #     eyeid.append (id)
        #     details.append (id + "_od" +"_right_"+ str (index))
        #     odOros.append ("od")
        #     indexs.append (index)
        #     CorpedRegion.append ("right")
        #     synechia.append (0)
        #     state.append (0)
        for index in range(12):
            eyeid.append (ids)
            details.append(id+"_left_"+str(index))
            odOros.append("os")
            indexs.append(index)
            CorpedRegion.append("left")
            synechia.append(0)
            state.append (0)
        for index in range(12):
            eyeid.append (ids)
            details.append(id+"_right_"+str(index))
            odOros.append("os")
            indexs.append(index)
            CorpedRegion.append("right")
            synechia.append(0)
            state.append (0)
    dicts = {"eyeId":eyeid, "odOros":odOros, "indexs":indexs, "CorpedRegion":CorpedRegion, "synechia":synechia, "details":details,"state":state}
    dict_df = pd.DataFrame(dicts)
    dict_df.to_csv("/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/wide_split/wide.csv")

    return 0



if __name__ == '__main__':
    path = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/wide_split/wide.txt"
    eyelist = list(i.strip("\n") for i in open(path).readlines())
    print(eyelist)
    creatCsv(eyelist)