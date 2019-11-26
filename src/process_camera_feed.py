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
# print (keras.version)
# print (tf.version())
# # from keras.models import Sequential
# # from keras.layers import Dense
# # from keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json
from matplotlib import pyplot as plt
import os

logging = True 

im_w = 1280
im_h = 720

class image_converter:

    def __init__(self):
        self.image_out = rospy.Publisher("/R1/image_out", Image, queue_size=1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/R1/pi_camera/image_raw", Image, self.callback)
        # self.model = self.loadmodel()
        self.graph = False

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        output, points = colormask_contour(cv_image)

        if points is not None:
            plateimg = trim_plate(cv_image, points)
            predictplate(plateimg)

        image_message = self.bridge.cv2_to_imgmsg(output, encoding="bgr8") #bgr8 or 8UC1
        self.image_out.publish( image_message )

def predictplate(plateimg):
    plate1 = (340,40)
    plate2 = (340,90)
    plate3 = (335,180)
    plate4 = (335,240)
    location = (180,180)

    im1 = feature_image(plateimg,plate1,1) # cropping
    im2 = feature_image(plateimg,plate2,1)
    im3 = feature_image(plateimg,plate3,1)
    im4 = feature_image(plateimg,plate4,1)
    im5 = feature_image(plateimg,location,2)

    # Are now loading the model each time to avoid the multithreading bug
    # This model was made with the simulated data. Will it work? Maybe
    # If it doesn't work, here are the options we have:
    # 1. Remake the fake data ( https://github.com/TyKeeling/ENPH353-competition/tree/master/enph353/enph353_gazebo/data_gen )
    # and then reimport to model. A subsection here is make MORE DATA but we're already training with 3610 images
    # 2. change the transformed image scaling and/or the window size (higher scaling resolution or larger CNN window)
    # This would mean changing the x_pipe and y_pipe in this file, as well as plate locations of course
    # 3. trial and error (find paper exmaples?) 
    # change model weights and biases, retrain, and import to real model to test. 
    json_file = open('/home/tyler/353_ws/src/license_process/src/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("/home/tyler/353_ws/src/license_process/src/model.h5")
    graph = tf.get_default_graph()
    loaded_model._make_predict_function()

    loaded_model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

    with graph.as_default():
        p1 = loaded_model.predict(im1)
        p2 = loaded_model.predict(im2)
        p3 = loaded_model.predict(im3)
        p4 = loaded_model.predict(im4)
        p5 = loaded_model.predict(im5)
        prediction = tochar(np.argmax(p1)) + tochar(np.argmax(p2)) + tochar(np.argmax(p3)) + tochar(np.argmax(p4)) + "_loc_" + str(np.argmax(p5))

        if logging:
            print("Plate Found: ", prediction)
            cv2.imwrite('guesses/' + prediction + '_' + str(rospy.get_rostime()) + '.png', plateimg)


# function that returns rectangular snippet of an image based on top xy coordinate and 
def feature_image(img, featureloc, scale=1, pipe_x=80, pipe_y=80):
    
    Ypoint = featureloc[0]
    if Ypoint + pipe_y*scale > im_h:
        raise Exception('section should not exceed im_h. The y endpoint was: {}'.format(Ypoint))

    Xpoint = featureloc[1]
    if Xpoint + pipe_x*scale > im_w:
        raise Exception('section should not exceed im_w. The x endpoint was: {}'.format(Xpoint))

    Ydown   = slice(Ypoint,Ypoint+pipe_y*scale, scale)
    Xacross = slice(Xpoint,Xpoint+pipe_x*scale, scale)

    if logging:
        pt1 = (Xpoint, Ypoint)
        pt2 = (Xpoint + pipe_x*scale, Ypoint + pipe_x*scale)
        cv2.rectangle(img,pt1,pt2,(0,255,0),1)

    newimg = img[Ydown,Xacross]
    newimg = np.expand_dims(newimg, axis=0)
    newimg = np.expand_dims(newimg, axis=4)

    return newimg

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



def trim_plate(img, points):
    border = 50
    res_x = 300
    res_y = int(res_x*1.5)
    plate_start = int(res_y*0.730)

    #This depends on how the plate is interpreted
    pts0 = np.float32([np.squeeze(points[0]), np.squeeze(points[1]), 
    np.squeeze(points[2]), np.squeeze(points[3])])

        # order the points according to pts2
    pts1 = np.float32([ mindist(pts0, 0, 0),
                        mindist(pts0, im_w, 0),
                        mindist(pts0, 0, im_h),
                        mindist(pts0, im_w, im_h) ])
    

    # print(pts1)

    for i in range(4): # this experiment shows that points are found but placed in no order
        cv2.circle(img, (int(pts1[i][0]),int(pts1[i][1])), 2, (255,255,255), (5*i+3))


    pts2 = np.float32([[0,0], [res_x,0], [0,plate_start], [res_x,plate_start]])
    pts2 += int(border/2)

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (res_x+border,res_y+border))

    im_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY) # wondering if some contrast would be nice here
    return im_gray


def colormask_contour(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV MASKS
    lower_red = np.array([0,0,90]) # white plates
    upper_red = np.array([255,60,220])

    lower_black = np.array([0,0,0])
    upper_black = np.array([255,10,20])
    
    colormask = cv2.inRange(hsv, lower_red, upper_red)     
    blackmask = cv2.inRange(hsv,lower_black,upper_black)
    bwmask = cv2.bitwise_or(colormask, blackmask)

    boxmask = purplemask(img, stripes=True)
    mask = cv2.bitwise_and(bwmask, boxmask)

    kernel = np.ones((2,2),np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # denoises the mask

    _, contours, h = cv2.findContours(opening, 1, 2)
    for cnt in contours: 
        approx = cv2.approxPolyDP(cnt,0.1*cv2.arcLength(cnt,True),True)
        if len(approx)==4:
            x,y,w,h = cv2.boundingRect(cnt)
            # h = int(h*1.37)
            if w > 60 and h > 80 and notEdges(x,y,w,h):
                return img, approx

    return img, None

def mindist(approx, x, y): #min of dist is min of dist^2
    mini = 5000 # should be fine as image is 1280pix in largest dimension
    i = 0
    for j in range(approx.shape[0]):
        dist = abs(x - approx[j][0]) + abs(y - approx[i][1]) # Assuming contours returns XY
        if dist < mini:  
            i = j 
            mini = dist

    return approx[i]

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

def purplemask(img, stripes=False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([120,30,30])
    upper_purple = np.array([130,255,255])
    purplemask = cv2.inRange(hsv, lower_purple, upper_purple)

    kernel = np.ones((2,2),np.uint8)
    opening = cv2.morphologyEx(purplemask, cv2.MORPH_OPEN, kernel)

    #Generate the stretch mask around the purple regions s.t. we can only find plates. 
    if stripes:
        overmask = np.zeros(purplemask.shape, np.dtype('uint8'))
        _, contours, h = cv2.findContours(opening, 1, 2)

        stretchbox = 100
        cutpix = 40 # to get rid of the small bottom part

        for cnt in contours:
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
