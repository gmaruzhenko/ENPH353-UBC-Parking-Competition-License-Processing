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

def main(args):
    image_raw = cv2.imread('../../../dataset/1574312167.41@DT24.png')
    cv2.imshow('test',image_raw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
