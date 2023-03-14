#Exporting required packages

import numpy as np
import cv2 as cv
import glob

#defining Checker Board Size and frame sie 

chessboard= (6,8)
framesize= (1440*1080)

#defining termination criteria for terminating subpixel iteration function
criteria= (cv.TERM_CRITERIA_EPS+ cv.TERM_CRITERIA_MAX_ITER,30,0.001)

#defining object points arrays

objp= np.zeros((chessboard[0]*chessboard[1],3), np.float32)
objp[:,:2]= np.mgrid[0:chessboard[0],0:chessboard[1]].T.reshape(-1,2)

# Creating blank arrays to store object points and image points from images
objpoints=[]
imgpoints=[]

#Exporting images for calibration

images= glob.glob('HW2/Cam_calib_images/*.jpg')

# Reading each image and making corners 

for image in images :
    print(image)
    img= cv.imread(image,cv.IMREAD_UNCHANGED)
    gray_image=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    #Finding chess board corners
    retval,corners= cv.findChessboardCorners(gray_image,chessboard,None)

    if retval== True:

        objpoints.append(objp)
        imgpoints.append(corners)
        #refining of corners via subpixels
        corners2= cv.cornerSubPix(gray_image,corners,(11,11),(-1,-1), criteria)

        #Drawing chess board corners in the image
        cv.drawChessboardCorners(img,chessboard,corners2,retval)
        cv.imshow('img',img)
        cv.waitKey(1000)
cv.destroyAllWindows()

# Calibrating Camera

ret,camera_matrix, dist, rvecs, tvecs= cv.calibrateCamera(objpoints, imgpoints, gray_image.shape[::-1], None, None)
print('camera calibrated: ', ret)
print('\nCamera Matrix = \n', camera_matrix)
print('\nDistortion Parameters = \n', dist)
print('\nRotation Vectors = \n', rvecs)
print('\nTranslation Vectors = \n', tvecs)

np.savez("Cameraparams",cameraMatrix=camera_matrix, dist = dist, rvecs=rvecs, tvecs=tvecs)