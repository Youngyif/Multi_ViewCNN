from shutil import copytree, ignore_patterns, copyfile
import os

# source = "/mnt/dataset/splited_Casia2/"
# destination = "/home/yangyifan/save/vertical"
# copytree (source, destination, ignore=ignore_patterns('*.jpg'))
wronglist=[]
for i, j, k in os.walk ("/home/datasets/CASIA2/zzz/"):
    for name in k:
        print (os.path.join(i,name))
        source = os.path.join(i,name)
        destination =os.path.join(i,name).replace("zzz", "splited_Casia2")
        print(destination)
        try:
            copyfile(source, destination)
        except:
            wronglist.append(destination)

print(len(wronglist), wronglist)