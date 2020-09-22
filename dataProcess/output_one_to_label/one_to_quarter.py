import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn import metrics

def getdf(path):
    df = pd.read_csv(path, index_col="details")
    return df
def getlist1(path):
    df = pd.read_csv(path)
    return df

def judge(df, df_medical, details):
    eyeid = details.split("_")[0]
    odos = details.split("_")[1]
    syne_label = df.loc[details,"details"]
    if int(syne_label)==1:
        return "synechia"
    else:
        if str(df_medical.loc[eyeid, odos.capitalize()[:2]+"state"]) == "1":#Odstate
            return "narrownotsyne"
        else :
            return "open"


def transfer(ids):
    splits = ids.split("_")
    return splits[0]+splits[1].capitalize()


def one_to_quater(details,dicts, proba):
    splits = details.strip("\n").split("_")
    ids = splits[0]+"_"+splits[1]
    lr = splits[-2]
    index = splits[-1]
    if lr == "left" and index in ["2","3","4"]:
        dicts[ids+"_2"].append(proba)
    elif lr=="right" and index in ["2","3","4"]:
        dicts[ids + "_0"].append(proba)
    elif (lr=="right" and index in ["0","1"]) or (lr=="left" and index=="5"):
        key = ids+"_1"
        dicts[key].append(proba)
    elif (lr=="left" and index in ["0","1"]) or (lr=="right" and index=="5"):
        dicts[ids + "_3"].append(proba)


def averagenum(num):
    nsum = 0
    for i in range(len(num)):

        nsum += num[i]
    return nsum / len(num)


def votenum(num):
    print("before",num)
    for i in range(len(num)):
        if num[i]>=0.5:
            num[i]=1
        else:
            num[i]=0
    print("after", num)
    print(sum(num) >= 2)
    return sum(num)>=2

def computeAUC(outputs, labels):
    pred = outputs
    y = labels
    fpr, tpr, thresholds = metrics.roc_curve (y, pred, pos_label=1)
    roc_auc = metrics.auc (fpr, tpr)
    if np.isnan (roc_auc):
        roc_auc = 0
    ####
    thresholdpred = []
    for i in pred:
        if i>=0.5:
            thresholdpred.append(1)
        elif i<0.5:
            thresholdpred.append(0)

    # pred[pred >= 0.5] = 1
    # pred[pred < 0.5] = 0

    # acc
    acc = metrics.accuracy_score(y, thresholdpred)
    # tn, fp, fn, tp
    tn, fp, fn, tp = metrics.confusion_matrix(y, thresholdpred).ravel()



    ####


    return roc_auc, fpr, tpr, tn, fp, fn, tp, acc
def getauc(dicts, flag):
    df_quarter = pd.read_csv("/home/yangyifan/code/multiviewCNN_quarter/multiViewCNN/Multi_ViewCNN/dataProcess/add_alone_clock/quarterlabelv3.csv", index_col="details")
    df_one = pd.read_csv("/home/yangyifan/code/multiviewCNN_quarter/multiViewCNN/Multi_ViewCNN/dataProcess/add_alone_clock/one_for_new_clock_labelv1.csv", index_col="details")
    labellist = []
    probalist = []
    if flag == "one":
        for i in dicts:
            # probalist.append(averagenum(dicts[i]))
            probalist.append(dicts[i])
            # labels = df_quarter.loc[i,"synechia"]
            labels = df_one.loc[i, "synechia"]
            labellist.append(labels)
        auc, fpr, tpr, tn, fp, fn, tp, acc = computeAUC(probalist, labellist)
        return auc, fpr, tpr, tn, fp, fn, tp, acc
    elif flag == "quarter":
        for i in dicts:
            # probalist.append(averagenum(dicts[i]))
            probalist.append(votenum(dicts[i]))
            # probalist.append(dicts[i])
            # labels = df_quarter.loc[i,"synechia"]
            labels = df_quarter.loc[i, "synechia"]
            labellist.append(labels)
        auc, fpr, tpr, tn, fp, fn, tp, acc = computeAUC(probalist, labellist)
        return auc, fpr, tpr, tn, fp, fn, tp, acc






def getonedict():
    path = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/probability_statistic_CONTRA.csv"
    # path_medical = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/medical_label/medicaldata.csv"
    dict = {}
    # df = getdf(path)
    df_label = getlist1(path)
    # df_medical = getdf(path_medical)
    detaillist = list(df_label["details"])
    probalist = list(df_label["probability"])
    for i in range(len(detaillist)):
        detail = detaillist[i]
        dict[detail] = probalist[i]
    return dict

def main1():
    path = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/probability_statistic_CONTRA.csv"
    # path_medical = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/medical_label/medicaldata.csv"
    dictone = getonedict()

    dicts_quarter = defaultdict(list)
    # df = getdf(path)
    df_label = getlist1(path)
    # df_medical = getdf(path_medical)
    detaillist = list(df_label["details"])
    probalist = list(df_label["probability"])
    for detail, proba in zip(detaillist,probalist):
        one_to_quater(detail, dicts_quarter, proba)
    # print(dicts_quarter)
    auc, fpr, tpr, tn, fp, fn, tp, acc = getauc(dicts_quarter,"quarter")
    print("Quarter_auc:{}  tn:{} fp:{} fn:{} tp:{} acc:{}".format(auc, tn, fp, fn, tp, acc))
    auc, fpr, tpr, tn, fp, fn, tp, acc = getauc(dictone,"one")
    print("One_auc:{}  tn:{} fp:{} fn:{} tp:{} acc:{}".format(auc, tn, fp, fn, tp, acc))


def main(openlistall, narrowlistall, synechialistall):
    path = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/probability_statistic.csv"
    # path_medical = "/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/medical_label/medicaldata.csv"

    dict={}
    # df = getdf(path)
    df_label = getlist1(path)
    # df_medical = getdf(path_medical)
    detaillist = list(df_label["details"])
    probalist = list(df_label["probability"])

    synechialist = []
    openlist = []
    narrow_not_synelist = []
    for i in range(len(detaillist)):
        print(i, detaillist[i])
        detail = detaillist[i]
        dict[detail] = probalist[i]
        temp = transfer(detail)
        if temp in synechialistall:
            synechialist.append(detail)
        elif temp in openlistall:
            openlist.append(detail)
        elif temp in narrowlistall:
            narrow_not_synelist.append(detail)
    print(len(set(narrow_not_synelist)),len(set(synechialist)),len(set(openlist)))
    print(set(narrow_not_synelist).intersection(set(openlist)))
    print(set(narrow_not_synelist).intersection(set(synechialist)))
    print(set(openlist).intersection(set(synechialist)))
    return openlist, narrow_not_synelist, synechialist, dict


def detailtransfer(details):
    splits = details.strip("\n").split("_")
    return splits[0]+splits[1].capitalize()
def main3():
    synechialist = np.load("/home/yangyifan/code/multiviewCNN_quarter/multiViewCNN/Multi_ViewCNN/dataProcess/add_alone_clock/synechia_exsist_list.npy")
    narrowlist = np.load("/home/yangyifan/code/multiviewCNN_quarter/multiViewCNN/Multi_ViewCNN/dataProcess/add_alone_clock/narrowNotSynechiaList.npy")
    openlist = list(open("/home/yangyifan/code/multiviewCNN_quarter/multiViewCNN/Multi_ViewCNN/dataProcess/add_alone_clock/openEye.txt").readlines())
    newopenlist=[]
    for i in range(len(openlist)):
        newopenlist.append(detailtransfer(openlist[i]))
    openlist, narrow_not_synelist, synechialist, dict= main(newopenlist, narrowlist, synechialist)
    proopen = []
    pronarrow = []
    prosyne = []
    for i in openlist:
        proopen.append(dict[i])
    for j in narrow_not_synelist:
        pronarrow.append(dict[j])
    for k in synechialist:
        prosyne.append(dict[k])
    dicts1 = {"open":openlist,"openpro":proopen}
    dicts2 = {"narrow_not_syne":narrow_not_synelist,"narrowpro":pronarrow}
    dicts3 = {"synechia": synechialist, "prosyne": prosyne}
    # df_csv1 = pd.DataFrame(dicts1)
    # df_csv1.to_csv('open_for_generate_image.csv')
    # df_csv2 = pd.DataFrame(dicts2)
    # df_csv2.to_csv('narrow_not_syne_for_generate_image.csv')
    # df_csv3 = pd.DataFrame(dicts3)
    # df_csv3.to_csv('synechia_for_generate_image.csv')

if __name__ == '__main__':
    # main()
    main1()
