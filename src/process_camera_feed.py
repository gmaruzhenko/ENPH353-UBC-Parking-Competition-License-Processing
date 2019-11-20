#!/usr/bin/env python
from __future__ import division

import sys
import time

import numpy as np

import cv2
import roslib
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Int16, String

import tensorflow as tf
from tensorflow import keras
# # from keras.models import Sequential
# # from keras.layers import Dense
# # from keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json
from matplotlib import pyplot as plt
import os

expected_error_max = 100

Kernel_size = 15
low_threshold = 75
high_threshold = 110
bwThresh = 100

class image_converter:

    def __init__(self):
        self.image_out = rospy.Publisher("/R1/image_out", Image, queue_size=1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/R1/pi_camera/image_raw", Image, self.callback)
        self.model = self.loadmodel()

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        output, (x,y,h,w) = colormask_contour(cv_image)

        if (x != 0 and y !=0):
            plateimg = trim_plate(cv_image,(x,y,h,w))
            plt.figure()
            plt.imshow(plateimg) 
            plt.show()
            self.predictplate(plateimg)

        image_message = self.bridge.cv2_to_imgmsg(output, encoding="bgr8") #bgr8 or 8UC1
        self.image_out.publish( image_message )
    
    def loadmodel(self):
        # model = keras.models.Sequential()
        # model.add(keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(80, 80, 1)))
        # model.add(keras.layers.MaxPooling2D((2, 2)))

        # model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
        # model.add(keras.layers.MaxPooling2D((2, 2)))

        # model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
        # model.add(keras.layers.MaxPooling2D((2, 2)))

        # model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
        # model.add(keras.layers.MaxPooling2D((2, 2)))

        # model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))

        # model.add(keras.layers.Flatten())
        # model.add(keras.layers.Dense(60, activation='relu'))
        # model.add(keras.layers.Dense(36, activation='softmax'))
        # model.load_weights("model.h5")
        # model._make_predict_function()

        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model.h5")
        loaded_model._make_predict_function()

        return loaded_model

    def predictplate(self,plateimg):
        pipe_x = 80
        pipe_y = 80

        plate1 = (340,40)
        plate2 = (340,90)
        plate3 = (335,180)
        plate4 = (335,240)
        location = (180,180)

        def feature_image(img, featureloc, scale=1):
            Ypoint = featureloc[0]
            Xpoint = featureloc[1]

            Ydown   = slice(Ypoint,Ypoint+pipe_y*scale, scale)
            Xacross = slice(Xpoint,Xpoint+pipe_x*scale, scale)

            newimg = img[Ydown,Xacross]
            np.expand_dims(newimg, axis=0)
            np.expand_dims(newimg, axis=4)
            return newimg

        im1 = feature_image(plateimg,plate1)
        im2 = feature_image(plateimg,plate2)
        im3 = feature_image(plateimg,plate3)
        im4 = feature_image(plateimg,plate4)
        im5 = feature_image(plateimg,location)

        p1 = self.model.predict(im1)
        p2 = self.model.predict(im2)
        p3 = self.model.predict(im3)
        p4 = self.model.predict(im4)
        p5 = self.model.predict(im5)

        prediction = str(p1) + str(p2) + str(p3) + str(p4) + str(p5)
        print(prediction)

def trim_plate(img, (x,y,h,w)):
    res_x = 300
    res_y = int(res_x*1.5)
    border = 50

    # pts1 = np.float32([     Need this only for the non rectangle images 
    #                 [x,y], 
    #                 [x+w,y], 
    #                 [x,y+h], 
    #                 [x+h,y+h]
    #                 ])
    
    # pts2 = np.float32([[0,0], [res_x,0], [0,res_y], [res_x,res_y]])
    # pts2 += int(border/2)
    
    # M = cv2.getPerspectiveTransform(pts1, pts2)
    
    # dst = cv2.warpPerspective(img, M, (res_x+border,res_y+border))

    crop = img[y:y+h,x:x+w]
    scale = cv2.resize(crop,(res_x,res_y))
    im_gray = cv2.cvtColor(scale, cv2.COLOR_BGR2GRAY) # wondering if some contrast would be nice here

    return im_gray

def toint(char):
    if char.isalpha():
        return ord(char) - 55 # A is 10, B is 11, ... Can use chr(65) to invert
    else:
        return int(char)

def tochar(inty):
    inty = int(inty)
    if inty < 10:
        return str(inty)
    else:
        return chr(55+inty) 

def notEdges(x, y, w, h, im_w=1280, im_h=720):
    if x <= 0:
        return False
    if y <= 0:
        return False
    if (x + w) >= im_w:
        return False
    if (y + h) >= im_h:
        return False
    return True

def colormask_contour(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV MASKS
    lower_red = np.array([0,0,90])
    upper_red = np.array([255,50,200])

    lower_black = np.array([0,0,0])
    upper_black = np.array([255,10,20])
    
    colormask = cv2.inRange(hsv, lower_red, upper_red) # actually white lol
    blackmask = cv2.inRange(hsv,lower_black,upper_black)
    bwmask = cv2.bitwise_or(colormask, blackmask)

    boxmask = purplemask(img, stripes=True)
    mask = cv2.bitwise_and(bwmask, boxmask)

    kernel = np.ones((2,2),np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # denoises the mask

    _, contours, h = cv2.findContours(opening, 1, 2)

    corners = (0,0,0,0)
    for cnt in contours: # hopefully only one rectangle is being output eek will fix later
        #cv2.drawContours(img,[cnt],0,(0,0,255),-1)
        approx = cv2.approxPolyDP(cnt,0.1*cv2.arcLength(cnt,True),True)
        if len(approx)==4:
            x,y,w,h = cv2.boundingRect(cnt)
            h = int(h*1.37)
            if w > 80 and h > 100 and notEdges(x,y,w,h):
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                corners = (x,y,h,w)

    return img, corners

def purplemask(img, stripes=False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([120,30,30])
    upper_purple = np.array([130,255,255])
    purplemask = cv2.inRange(hsv, lower_purple, upper_purple)

    kernel = np.ones((2,2),np.uint8)
    opening = cv2.morphologyEx(purplemask, cv2.MORPH_OPEN, kernel)

    #Generate the janky mask around the purple values s.t. we can only find plates. 
    if stripes:
        overmask = np.zeros(purplemask.shape, np.dtype('uint8'))
        _, contours, h = cv2.findContours(opening, 1, 2)

        stretchbox = 100
        cutpix = 40 # hopefully to get rid of the bottom part of the thing

        for cnt in contours:
            #cv2.drawContours(img,[cnt],0,(0,0,255),-1)
            x,y,w,h = cv2.boundingRect(cnt)
            x -= stretchbox
            w += 2*stretchbox
            h -= cutpix
            cv2.rectangle(overmask,(x,y),(x+w,y+h),(255), cv2.FILLED)

        return overmask
    return opening

def main(args):
    rospy.init_node('image_converter', anonymous=True)
    ic = image_converter()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
