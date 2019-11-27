# Here we go
# Over 2000 real data images. What to do??!?
# Be suspicious of DD00

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import re
import os

from matplotlib import pyplot as plt
import cv2

#First part just gets the filenames and such ready:
folder = os.getcwd() + "/../../../guesses/"
outfolder = os.getcwd() + "/../../../labelled/"

onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
filenames = [] # the image filename 
labels = []     # the label

for _file in onlyfiles: # file names
    filenames.append(_file)
    
print(str(len(filenames)) + " names loaded, lets go!")

laststring = "XX00_5"

i = 0 
while i < range(int(len(filenames))):
    send = True
    location = folder + "/" + filenames[i]
    
    im = cv2.imread(location)
    plt.figure()
    plt.title(laststring)
    plt.imshow(im) 
    plt.show()
    
    resp = input("What should that title be? \n[c]orrect, [z]nope, [b]ack or a 5 letter string please:")
    
    if resp == "c":
        outstring = laststring
        i += 1 
        
    elif resp == "z":
        print("skipping")
        send = False
        i+= 1
        
    elif resp == "b":
        i-=1
        send = False
        print("go delete your _up")
        
    elif len(resp) == 5:
        outstring = resp
        
    else:
        send = False
        print("Try again:")
        
    laststring = outstring
    if send:
        outlocation = outfolder + "/" + outstring + ".png"
        
    print("moving to image " + str(i))
    

