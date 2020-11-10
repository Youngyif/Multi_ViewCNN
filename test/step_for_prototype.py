import torch
import random
def step(t1,t2): ##each t represent feature vector of a AS-OCT sequence thus its dimension is 21, the same as number of slices on AS-OCT sequence
    step=random.randint(1,5)
    list=[]
    len=t1.size(1)
    index=0
    while index<len:
        t1[:, ]
