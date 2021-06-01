import os
import cv2
import re


dirname1 = "0class"
os.mkdir(dirname1)
dirname2 = "90class"
os.mkdir(dirname2)
dirname3 = "180class"
os.mkdir(dirname3)
fpath = "/home/mfusco/3classes"
j=0
i=0
f = open('matched090.txt','r')
k = open('matched0180.txt','r')


with f as fp : lines1 = fp.read().splitlines()
with k as kp : lines2 = kp.read().splitlines()

for line in lines1:
    j=j+1
    val = line.split("\t")
    #print(val, "line", j)
    for lin in lines2:
        vall = lin.split("\t")
        #print(vall)
        if val[0]==vall[0] and val[1]==vall[1]:

        for image_name in os.listdir(fpath + "/" +  "0"):
                    grainid = re.findall ("g(\d+).png", image_name)
                    viewid = re.findall("v(\d+)", image_name)
            
                    if grainid[0] == val[1] and viewid[0] == val[0]:
                       for image_name2 in os.listdir(fpath + "/" + "90"):
                           grainid2 = re.findall("g(\d+).png", image_name2)
                           viewid2 = re.findall("v(\d+)", image_name2)
                           if grainid2[0] == val[3] and viewid2[0] == val[2]:
                               #i = i+1
                               #img1 = cv2.imread(fpath + "/0/" + image_name)
                               #img2 = cv2.imread(fpath + "/90/" + image_name2)
                               #nome1 = f"{i}_v{val[0]}_g{val[1]}.png"
                               #nome2 = f"{i}_v{val[2]}_g{val[3]}.png"
                               #cv2.imwrite(os.path.join(dirname1, nome1), img1)
                               #cv2.imwrite(os.path.join(dirname2, nome2), img2)
                              # print(i)
                            #else:
                                #print("no correscpondence found!") 
                                for image_name3 in os.listdir(fpath + "/" + "180"):
                                        grainid3 = re.findall("g(\d+).png", image_name3)
                                        viewid3 = re.findall("v(\d+)", image_name3)
                                        if grainid3[0] == vall[3] and viewid3[0] == vall[2]:
                                           i = i+1
                                           img1 = cv2.imread(fpath + "/0/" + image_name)
                                           img2 = cv2.imread(fpath + "/90/" + image_name2)
                                           img3 = cv2.imread(fpath + "/180/" + image_name3)
                                           nome1 = f"{i}_v{val[0]}_g{val[1]}.png"
                                           nome2 = f"{i}_v{val[2]}_g{val[3]}.png"
                                           nome3 = f"{i}_v{vall[2]}_g{vall[3]}.png"
                                           cv2.imwrite(os.path.join(dirname1, nome1), img1)
                                           cv2.imwrite(os.path.join(dirname2, nome2), img2)
                                           cv2.imwrite(os.path.join(dirname3, nome3), img3)
                                           print(i)
        
f.close()
k.close()
