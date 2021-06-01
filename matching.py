import os
# this script read the name of the matched image from file txt and search for the corresponding images in fpath then copy them in dirname1 and 2


import cv2
import re

dirname1 = "0grain"
os.mkdir(dirname1)
dirname2 = "90grain"
os.mkdir(dirname2)
fpath = "/home/mfusco/singledataset"
j=0
i=0
f = open('matched.txt','r')


with f as fp : lines = fp.read().splitlines()

for line in lines:
    j=j+1
    val = line.split("\t")
    #print(val, " line ", j)
    for image_name in os.listdir(fpath + "/" + "0"):
            grainid = re.findall ("g(\d+).png", image_name)
            viewid = re.findall("v(\d+)", image_name)
            
            if grainid[0] == val[1] and viewid[0] == val[0]:
               for image_name2 in os.listdir(fpath + "/" + "90"):
                   grainid2 = re.findall("g(\d+).png", image_name2)
                   viewid2 = re.findall("v(\d+)", image_name2)
                   if grainid2[0] == val[3] and viewid2[0] == val[2]:
                       i = i+1
                       img1 = cv2.imread(fpath + "/0/" + image_name)
                       img2 = cv2.imread(fpath + "/90/" + image_name2)
                       nome1 = f"{i}_v{val[0]}_g{val[1]}.png"
                       nome2 = f"{i}_v{val[2]}_g{val[3]}.png"
                       cv2.imwrite(os.path.join(dirname1, nome1), img1)
                       cv2.imwrite(os.path.join(dirname2, nome2), img2)
                       print(i)
                    #else:
                        #print("no correscpondence found!") 

f.close()
