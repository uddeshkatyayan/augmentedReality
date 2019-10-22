import numpy as np
import cv2
import glob
import sys
import math
import os
from objLoader import *

#from utils.kp_tools import kp_list_2_opencv_kp_list

MIN_MATCHES = 150
def projection_matrix(camera_parameters, homography):
    
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    
    rot_3 = np.cross(rot_1, rot_2)
    
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    
    return np.dot(camera_parameters, projection)

def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img


if __name__ == "__main__" :

    intrinsic_params = []
    with open('./calibration_parameter.txt') as f:
        for line in f.readlines():
            col = [float(i) for i in line.split()]
            intrinsic_params.append(col)

    obj = OBJ(os.path.join('./', 'models/fox.obj'), swapyz=True)

    capx = cv2.VideoCapture(-1)

    frames = []
    persistent_img = np.ndarray([])

    model_path = sys.argv[2]
    model_name = glob.glob(model_path+"/*")
    model = cv2.imread(model_name[0],0)
    print(model_name[0])

    while(True):

        ret, frame = capx.read()

        if cv2.waitKey(1) & 0xFF == ord('a'):
            break
        cap = frame

        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        kp_model, des_model = orb.detectAndCompute(model, None)
        kp_frame, des_frame = orb.detectAndCompute(cap, None)
        matches = bf.match(des_model, des_frame)
        print(len(matches))

        if len(matches) < MIN_MATCHES:
            cv2.imshow('output',persistent_img)
            continue

        src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = model.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        dst = cv2.perspectiveTransform(pts, M)
        proj_mat = projection_matrix(intrinsic_params,M)

        finalImg = render(cap, obj, proj_mat, model, False)
        persistent_img = finalImg

        cv2.imshow('output',finalImg)

