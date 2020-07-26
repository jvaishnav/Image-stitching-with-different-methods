#!/usr/bin/env python


from __future__ import print_function
#from basicmotiondetector import BasicMotionDetector
from panorama import Stitcher
#from imutils.video import VideoStream
import numpy as np
import datetime
import imutils
import time
import cv2
import pyzed.sl as sl
# initialize the video streams and allow them to warmup
print("[INFO] starting cameras...")
'''video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH,2560)
video.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
video.set(cv2.CAP_PROP_FPS, 15)
time.sleep(2.0)'''
zed = sl.Camera()

        # Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
init_params.camera_resolution = sl.RESOLUTION.HD720
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
   exit(1)

runtime_parameters = sl.RuntimeParameters()
runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD
runtime_parameters.confidence_threshold = 100
runtime_parameters.textureness_confidence_threshold = 100

        
        #image = sl.Mat()
image_size = zed.get_camera_information().camera_resolution
image_size.width = image_size.width 
image_size.height = image_size.height 
mirror_ref = sl.Transform()
mirror_ref.set_translation(sl.Translation(2.75, 4.0, 0))
image = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
depth = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
point_cloud = sl.Mat()
tr_np = mirror_ref.m
stitcher = Stitcher()
key = ' '
#print('4')
while key != 113:
 if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
    zed.retrieve_image(image, sl.VIEW.SIDE_BY_SIDE, sl.MEM.CPU)
    #zed.retrieve_image(image, sl.VIEW.LEFT) 
    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
    depth_image_ocv = depth.get_data()
    img = image.get_data()

# initialize the image stitcher, motion detector, and total
# number of frames read
#stitcher = Stitcher()
#motion = BasicMotionDetector(minArea=500)
    total = 0

# loop over frames from the video streams
#while True:
	#ret,frame = video.read()
        #print("frame:", frame.shape)


    #left_right_image = np.split(frame, 2, axis=1) 
	# grab the frames from their respective video streams
    #left = left_right_image[0]
    left = img[0:720, 0:1280]
        #print("left:", left.shape)
    left = cv2.resize(left, (640,360))
    #left = cv2.cvtColor(left,cv2.COLOR_BGR2GRAY)

    #right = left_right_image[1]
    right = img[0:720, 1280:2560]
        #print("right:", right.shape)
    right = cv2.resize(right, (640,360))
    #using a binary string based descriptor ORB (oriented fast and rotated brief) a link to opencv ORB: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html

#initiate ORB detector
    orb = cv2.ORB_create() 

    # "kp*" term finds the keypoints using FAST keypoint detector and "des*" term is found by using the BRIEF descriptors (you can read more about it from the above mentioned link)
    kp1, des1 = orb.detectAndCompute(left, None)
    des1 = np.float32(des1)
    kp2, des2 = orb.detectAndCompute(right, None)
    des2 = np.float32(des2)
    #print(type(kp1))


#FLANN Based Matcher 
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
       if m.distance < 0.7*n.distance:
          good.append(m)
       #print('good =', len(good))

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
       print ("Not enough matches are found - %d/%d")
       matchesMask = None


    


    dst = cv2.warpPerspective(left,M,(right.shape[1] + left.shape[1], right.shape[0]))
    dst[0:right.shape[0],0:left.shape[1]] = left 
    cropped_image = dst[0:720, 0:680]

    cv2.imshow("left", left)
    cv2.imshow("right", right)
    cv2.imshow("original_image_stitched", cropped_image)
    #cv2.imshow("Matching result", matching_result)
    if cv2.waitKey(30) >= 0 :
        break

    #img3 = cv2.drawMatches(left,kp1,right,kp2,good,None,**draw_params)



