from shutil import copytree, ignore_patterns
import os

# source = "/mnt/dataset/splited_Casia2/"
# destination = "/home/yangyifan/save/vertical"
# copytree (source, destination, ignore=ignore_patterns('*.jpg'))
for i,j,k in os.walk("/home/yangyifan/save/vertical"):
    print(i,j,k)

