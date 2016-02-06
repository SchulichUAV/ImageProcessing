import numpy as np
import cv2
import glob

print("i loaded the libraries")
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.jpg')

for fname in images:
    img = cv2.imread(fname)
    smol = cv2.resize(img, (0,0), fx = 0.4, fy = 0.4)
    gray = cv2.cvtColor(smol,cv2.COLOR_BGR2GRAY)
    grid = (7,7)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, grid,None)

    # If found, add object points, image points (after refining them)
    
    if ret == True:
        print ("here's ret")
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

            # Draw and display the corners
        cv2.drawChessboardCorners(smol, grid, corners,ret)
        cv2.imshow('thesight',smol)
        cv2.waitKey(0)


cv2.destroyAllWindows()