import numpy as np
import cv2
import glob


def calibrateCamera(image_folder):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('./'+image_folder+'/'+'*.jpg')

    print(images)
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow('input',img)
        cv2.waitKey(500)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,7),None)
        print(ret)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7,7), corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    #callibration
    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

if __name__ == "__main__" :
    ret, mtx, dist, rvecs, tvecs = calibrateCamera("images")
#    frames = saveImageFromWebcam()
    f = open("calibration_parameter.txt","w+")    
    for i in range(len(mtx)):
        for j in range(len(mtx[0])):
            f.write("%f " % mtx[i][j])
        f.write("\n")    
    f.close()
