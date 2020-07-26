#!/usr/bin/env python

#import uvcvideo
#modprobe uvcvideo nodrop=1 timeout=5000 quirks=0x80
#import rospy
from std_msgs.msg import String
import cv2
import numpy as np
import matplotlib.pyplot as plt

'''while True:
    try:
        # Note: Python 2.x users should use raw_input, the equivalent of 3.x's input
        print('preferred resolution options are mentioned as below \n 1) = 1344 x 376 \n 2) = 2560 x 720 \n 3) = 3840 x 1080 \n 4) = 4416 x 1242')
        resolution = int(input("Please enter your option from the above mentioned: "))
    except ValueError:
        print("Sorry, I didn't understand that.")
        #better try again... Return to the start of the loop
        continue
    else:
        #age was successfully parsed!
        #we're ready to exit the loop.
        break
if resolution == 1:
       x = 1344
       y = 376
       print ('x=' , x , 'y=' , y)

elif resolution == 2:
       x = 2560
       y = 720
       print ('x=' , x , 'y=' , y)

elif resolution == 3:
       x = 3840
       y = 1080
       print ('x=' , x , 'y=' , y)

elif resolution == 4:
       x = 4416
       y = 1242
       print ('x=' , x , 'y=' , y)'''




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




#initiate ORB detector
    orb = cv2.ORB_create() 

    # "kp*" term finds the keypoints using FAST keypoint detector and "des*" term is found by using the BRIEF descriptors (you can read more about it from the above mentioned link)
    kp1, des1 = orb.detectAndCompute(left, None)
    kp2, des2 = orb.detectAndCompute(right, None)
    #print(type(kp1))



#create BFMatcher object
    #bf = cv2.BFMatcher()                         #with default params

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #could use norm hamming for distance measurement between and crosscheck is switched on for better results



# Match descriptors.
    #matches = bf.knnMatch(des1, des2, k = 2)            #tries to find the best matches
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    print ('matches=',len(matches))
    
    good = []
    #print(len(good))

    #applying NN ratio test
    for m in matches:
     if m.distance > 0.7:
        good.append(m)
    print(len(good))

    '''for m, n in matches:
        if m.distance < 0.7 * n.distance:
           good.append(m)'''
    matching_result = cv2.drawMatches(left,kp1,right,kp2,matches[:1000], None, flags=2)

    
    


    if len(good) > 4:           # if enough matches are found they are passed to find the perspective transfoemation
       src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
       dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
       
       #Find homography and use RANSAC to align the images
       M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
       matchesMask = mask.ravel().tolist()
       print('M =', M)
       print('left.shape=', len(left.shape), left.shape)

       h,w, _ = left.shape
       print('h=',h,'w=',w)
       pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
       print('pts=', len(pts), pts.shape)
       dst = cv2.perspectiveTransform(pts, M)
       print('dst=', len(dst), dst.shape)
       right = cv2.polylines(right,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA)
    #cv2.imshow("original_image_overlapping.jpg", right)
    else:
       print("Not enough matches are found - %d/%d", (len(good)))

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    
    
    
    dst = cv2.warpPerspective(left,M,(right.shape[1] + left.shape[1], right.shape[0]))
    dst[0:right.shape[0],0:left.shape[1]] = right
    cv2.imshow("Matching result", matching_result)
    cv2.imshow("original_image_stitched", dst)
    if cv2.waitKey(30) >= 0 :
        break
    

