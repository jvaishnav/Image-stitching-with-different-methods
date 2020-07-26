#!/usr/bin/env python


import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from math import sqrt



cap = cv2.VideoCapture(0)
if cap.isOpened() == 0:
    exit(-1)

# Set the video resolution to HD720 (2560*720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap. set(cv2.CAP_PROP_FPS, 60)



#stitcher = cv2.Stitcher.create(False)
while True :
    # Get a new frame from camera
    retval, frame = cap.read()
    # Extract left and right images from side-by-side
    #frame = np.uint8(frame)
    left_right_image = np.split(frame, 2, axis=1)
    # Display images
    left = left_right_image[1]
    left = cv2.resize(left, (640,360))
    #left = cv2.cvtColor(left,cv2.COLOR_BGR2GRAY)

    right = left_right_image[0]
    right = cv2.resize(right, (640,360))
    #right = cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)
    
    #using a binary string based descriptor ORB (oriented fast and rotated brief) a link to opencv ORB: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html
    



    
    

#initiate AKAZE detector
    akaze = cv2.AKAZE_create()

    # "kp*" term finds the keypoints using FAST keypoint detector and "des*" term is found by using the BRIEF descriptors (you can read more about it from the above mentioned link)
    kp1, des1 = akaze.detectAndCompute(left, None)
    
    kp2, des2 = akaze.detectAndCompute(right, None)
    
    #print(type(kp1))


    '''matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    nn_matches = matcher.knnMatch(des1, des2, 2)'''

#    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf = cv2.BFMatcher()
    nn_matches = bf.match(des1,des2)

    good = []
    nn_match_ratio = 0.7 # Nearest neighbor matching ratio
    '''for m, n in nn_matches:
        if m.distance < nn_match_ratio * n.distance:
           good.append(m)
        print(len(good))'''


    for i, m in enumerate(nn_matches):
        if i < len(nn_matches) - 1 and m.distance < 0.7 * nn_matches[i+1].distance:
           good.append(m)
        print(len(good))
    #matching_result = cv2.drawMatches(left,kp1,right,kp2,matches, None, **draw_params)



    if len(good) > 4:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w,_ = left.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
        #img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
       print "Not enough matches are found - %d/%d" % (len(good),4)
       matchesMask = None


    


    dst = cv2.warpPerspective(left,M,(right.shape[1] + left.shape[1], right.shape[0]))
    dst[0:right.shape[0],0:left.shape[1]] = right
    #cropped_image = dst[0:720, 0:680]

    #cv2.imshow("left", left)
    #cv2.imshow("right", right)
    cv2.imshow("original_image_stitched", dst)
    #cv2.imshow("original_image_stitched", cropped_image)
    #cv2.imshow("Matching result", matching_result)
    if cv2.waitKey(30) >= 0 :
        break

    #img3 = cv2.drawMatches(left,kp1,right,kp2,good,None,**draw_params)'''



