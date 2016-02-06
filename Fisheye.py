import numpy as np
import cv2
import glob

print("i loaded the libraries")
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7 * 7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.jpg')

for fname in images:
    img = cv2.imread(fname)
    smol = cv2.resize(img, (0,0), fx = 1, fy = 1)
    gray = cv2.cvtColor(smol,cv2.COLOR_BGR2GRAY)
    grid = (7,7)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, grid,None)
    print ("Corners: ", len(corners))

    # If found, add object points, image points (after refining them)
    
    if ret == True:
        print ("Found Chessboard")
        objpoints.append(objp)
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

            # Draw and display the corners
        cv2.drawChessboardCorners(smol, grid, corners,ret)
        cv2.imshow('thesight',smol)
    
    print ("Calibrate? Y / N")



    if cv2.waitKey(0) == ord('y'):
        print ("Calibrating.")
        test = gray.shape[::-1]
        print ("objpoints: ", objpoints, len(objpoints[0]))
        print ("imgpoints: ", imgpoints, len(imgpoints[0]))
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
       
        h, w = gray.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        print (newcameramtx,roi)


        if 1:
            dst = cv2.undistort(smol, mtx, dist, None, newcameramtx)
        else:
            mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
            dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
        
        x,y,h,w = roi
        cv2.imshow("undistort",dst)

        print ("Save Photo? Y /N")
        if cv2.waitKey(0) == ord('y'):    
            
            cv2.imwrite('calibresult.png',dst)
        cv2.destroyAllWindows()
cv2.destroyAllWindows()