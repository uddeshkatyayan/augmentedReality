import numpy as np
import cv2
import glob
import sys
import math
import os
from objLoader import *

#from utils.kp_tools import kp_list_2_opencv_kp_list
N = 25
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

    obj = OBJ(os.path.join('./models/', 'fox.obj'), swapyz=True)

    capx = cv2.VideoCapture(-1)

    frames = []
    persistent_img = np.ndarray([])

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    model_path = sys.argv[2]
    model_name = glob.glob(model_path+"/*")
    print(model_name)
    assert len(model_name) > 1, "There must be at least 2 visual markers present in the directory."
    models = [cv2.imread(m,0) for m in model_name[:2]]
    model = models[0]

    print("identifying",model_name[0])
    found = False
    result = []

    while not found:
        ret, frame = capx.read()
        #cv2.imshow('input',frame)
        #if cv2.waitKey(1) & 0xFF == ord('a'):
            #break

        cap = frame
        kp_frame, des_frame = orb.detectAndCompute(cap, None)

        for idx,model in enumerate(models[1:]):
            kp_model, des_model = orb.detectAndCompute(model, None)
            matches = bf.match(des_model, des_frame)
            print(idx, len(matches))
            if len(matches) < MIN_MATCHES:
            	found = False
            	break
            result = frame
            found = True

    wall = models[0]
    base = models[1]
    kp_frame, des_frame = orb.detectAndCompute(cap, None)
    kp_base, des_base = orb.detectAndCompute(base, None)
    kp_wall, des_wall = orb.detectAndCompute(wall, None)
    matches_wall = bf.match(des_wall, des_frame)
    matches_base = bf.match(des_base, des_frame)
    frame_pts_wall = np.float32([kp_frame[m.trainIdx].pt for m in matches_wall]).reshape(-1, 1, 2)
    frame_pts_base = np.float32([kp_frame[m.trainIdx].pt for m in matches_base]).reshape(-1, 1, 2)
    wall_pts = np.float32([kp_wall[m.queryIdx].pt for m in matches_wall]).reshape(-1, 1, 2)
    base_pts = np.float32([kp_base[m.queryIdx].pt for m in matches_base]).reshape(-1, 1, 2)
    M_wall, mask = cv2.findHomography(wall_pts, frame_pts_wall, cv2.RANSAC, 5.0)
    M_base, mask = cv2.findHomography(base_pts, frame_pts_base, cv2.RANSAC, 5.0)

    h, w = wall.shape
    pts_wall = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst_wall = cv2.perspectiveTransform(pts_wall, M_wall)
    print(dst_wall.shape)
    proj_mat_wall = projection_matrix(intrinsic_params,M_wall)

    h, w = base.shape
    pts_base = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst_base = cv2.perspectiveTransform(pts_base, (M_base))
    diff = (dst_wall-dst_base)
    dx = [x[0][0] for x in diff]
    dy = [x[0][1] for x in diff]

    for i in range(N):
        t = [i*sum(dx)/4*N,i*sum(dy)/4*N]
        Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
        print(Ht)
        dst_base = cv2.perspectiveTransform(pts_base, Ht.dot(M_base))
        proj_mat_base = projection_matrix(intrinsic_params, Ht.dot(M_base))
        finalImg = render(cap, obj, proj_mat_base, model, False)
        cv2.imshow('output',finalImg)
        cv2.waitKey(100)


    #finalImg = render(cap, obj, proj_mat_wall, model, False)
    proj_mat_base = projection_matrix(intrinsic_params, (M_base))
    finalImg = render(finalImg, obj, proj_mat_base, model, False)

    cv2.imshow('output',finalImg)
    cv2.waitKey(0)
