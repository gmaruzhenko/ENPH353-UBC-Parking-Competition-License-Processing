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
from matplotlib import pyplot as plt

im_w = 1280
im_h = 720


def trim_plate(img, points):
    border = 0
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
    

    print(pts1)

    for i in range(4): # this experiment shows that points are found but placed in no order
        cv2.circle(img, (int(pts1[i][0]),int(pts1[i][1])), 2, (255,255,255), (5*i+3))


    pts2 = np.float32([[0,0], [res_x,0], [0,plate_start], [res_x,plate_start]])
    pts2 += int(border/2)

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (res_x+border,res_y+border))

    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # wondering if some contrast would be nice here
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

    print (approx[i])
    print (mini)
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
    image_raw = cv2.imread('../../../dataset/1574312167.41@DT24.png')
    cv2.imshow('test',image_raw)
    cv2.waitKey(0)
   

    output, points = colormask_contour(image_raw)
    trimmed_image = trim_plate(image_raw,points)
    cv2.imshow('timmed',trimmed_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
if __name__ == '__main__':
    main(sys.argv)
