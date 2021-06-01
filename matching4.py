import os
import cv2
import re


dirname1 = "0cl"
os.mkdir(dirname1)
dirname2 = "90cl"
os.mkdir(dirname2)
dirname3 = "180cl"
os.mkdir(dirname3)
dirname4 = "270cl"
os.mkdir(dirname4)
fpath = "/home/mfusco/4classes"
j=0
i=0
f = open('matched090.txt','r')
k = open('matched0180.txt','r')
y = open('matched0270.txt','r')

with f as fp : lines1 = fp.read().splitlines()
with k as kp : lines2 = kp.read().splitlines()
with y as kp : lines3 = yp.read().splitlines()

for line in lines1:
    #j=j+1
    val = line.split("\t")
    #print(val, "line", j)
    for lin in lines2:
        vall = lin.split("\t")
        #print(vall)
        if val[0]==vall[0] and val[1]==vall[1]:
           for li in lines3:
             va = li.split("\t")
             #print(vall)
             if val[0]==vall[0]==va[0] and val[1]==vall[1]==va[1]:

                        for image_name in os.listdir(fpath + "/" +  "0"):
                                 grainid = re.findall ("g(\d+).png", image_name)
                                 viewid = re.findall("v(\d+)", image_name)
                                 if grainid[0] == val[1] and viewid[0] == val[0]:

                                        for image_name2 in os.listdir(fpath + "/" + "90"):
                                                grainid2 = re.findall("g(\d+).png", image_name2)
                                                viewid2 = re.findall("v(\d+)", image_name2)
                                                if grainid2[0] == val[3] and viewid2[0] == val[2]:

                                                     for image_name3 in os.listdir(fpath + "/" + "180"):
                                                        grainid3 = re.findall("g(\d+).png", image_name3)
                                                        viewid3 = re.findall("v(\d+)", image_name3)
                                                        if grainid3[0] == vall[3] and viewid3[0] == vall[2]:


                                                           for image_name4 in os.listdir(fpath + "/" + "270"):
                                                               grainid4 = re.findall("g(\d+).png", image_name4)
                                                               viewid4 = re.findall("v(\d+)", image_name4)
                                                               if grainid4[0] == va[3] and viewid4[0] == va[2]:

                                                                   i = i+1
                                                                   img1 = cv2.imread(fpath + "/0/" + image_name)
                                                                   img2 = cv2.imread(fpath + "/90/" + image_name2)
                                                                   img3 = cv2.imread(fpath + "/180/" + image_name3)
                                                                   img4 = cv2.imread(fpath + "/270/" + image_name4)
                                                                   nome1 = f"{i}_v{val[0]}_g{val[1]}.png"
                                                                   nome2 = f"{i}_v{val[2]}_g{val[3]}.png"
                                                                   nome3 = f"{i}_v{vall[2]}_g{vall[3]}.png"
                                                                   nome4 = f"{i}_v{va[2]}_g{va[3]}.png"
                                                                   cv2.imwrite(os.path.join(dirname1, nome1), img1)
                                                                   cv2.imwrite(os.path.join(dirname2, nome2), img2)
                                                                   cv2.imwrite(os.path.join(dirname3, nome3), img3)
                                                                   cv2.imwrite(os.path.join(dirname4, nome4), img4)
                                                                   print(i)
        
f.close()
k.close()
y.close()
