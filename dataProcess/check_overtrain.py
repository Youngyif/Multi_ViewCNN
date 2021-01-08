f=open("/mnt/cephfs/home/yangyifan/yangyifan/232/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/split_bjdata/trainvallist/overtrain/train_bj.txt")
txt=list(f.readlines())
txt_set=set(txt)
from collections import Counter
# print(Counter(txt))
temp=0
for key,count in Counter(txt).items():
    if count>1:
        temp+=1
        print(key, count)
    if count!=3:
        print("what happened",key)

print(temp)