#! /usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image
import cv2
import math
from cv_bridge import CvBridge,CvBridgeError
import message_filters
import os
import sys
import argparse
import torch
import tf
import numpy as np
import open3d as o3d
from PIL import Image as Ig
from Anygrasp_sdk.srv import rpypose, rpyposeRequest, rpyposeResponse


bridge = CvBridge()

path = os.path.abspath(".")
sys.path.insert(0,path + "/src/Anygrasp_sdk/scripts")

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=False, default='log/checkpoint_detection.tar' , help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.085, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.023, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', default=True, help='Output top-down grasps')
parser.add_argument('--debug', action='store_true', default=True, help='Enable visualization')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(theta)

# 控制目标夹爪的夹角，angle的单位是度
def angle_control(gg,angle): 
    ggs = GraspGroup()
    ggss = GraspGroup()
    for g in gg:
        g_x_vector = g.rotation_matrix[:, 0]
        z_vector = np.array([0, 0, 1])
        if angle_between_vectors(g_x_vector,z_vector) <=angle:
            ggs.add(g)
        # print(angle_between_vectors(g_x_vector,z_vector))
    # gg = ggs
    for g in ggs:
        g_x_vector = g.rotation_matrix[:, 1]
        x_vector = np.array([1, 0, 0])
        if angle_between_vectors(g_x_vector,x_vector) <=angle:
            ggss.add(g)
        print(angle_between_vectors(g_x_vector,x_vector))
    return ggss

def kinect_subscriber():
    
    # color = rospy.wait_for_message('/rgb/image_raw',Image,timeout=None)
    # depth = rospy.wait_for_message('/depth_to_rgb/image_raw',Image,timeout=None)
    # cv_rgb = bridge.imgmsg_to_cv2(color, "bgr8")
    # # cv_rgb = cv2.cvtColor(cv_rgb,cv2.COLOR_BGR2RGB)
    # cv_depth = bridge.imgmsg_to_cv2(depth, "16UC1")


    # img = cv2.bitwise_not(cv_rgb)
    
    # cv2.imwrite("/home/ani/Desktop/example_data/" + str(rospy.Time.now().to_sec()) + "color.png",cv_rgb)
    # cv2.imwrite("/home/ani/Desktop/example_data/" + str(rospy.Time.now().to_sec()) + "depth.png",cv_depth)
    # cv_rgb = Ig.open('/home/ani/Desktop/dynamic/ob4/color_034.png')
    # cv_depth = Ig.open('/home/ani/Desktop/dynamic/ob4/depth_034.png')
    cv_rgb = Ig.open('/home/ani/Desktop/example_data/detection/color2.png')
    cv_depth = Ig.open('/home/ani/Desktop/example_data/detection/depth2.png')

    colors = np.array(cv_rgb, dtype=np.float32) / 255.0
    depths = np.array(cv_depth)
    
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # get camera intrinsics
    fx, fy = 969.750743, 967.848535
    cx, cy = 1031.790933, 771.076245
    xwidth , yheight = 2048, 1536
    k1, k2 = 0.074274, -0.035373   # 径向畸变参数，需要消除畸变
    scale = 1000.0   # kinect深度图的深度的单位mm和此刻使用的单位m的转换关系

    # set workspace
    xmin, xmax = -0.22, 0.22
    ymin, ymax = -0.22, 0.22
    zmin, zmax = 0.0, 1.0
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    # get point cloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap = xmap - cx
    ymap = ymap - cy
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depths / scale
    points_x = xmap / fx * points_z
    points_y = ymap / fy * points_z

    # print(points_x)
    # remove distortion 消除畸变
    for i in range(yheight):
        for j in range(xwidth):
            r = math.sqrt(pow(points_x[i][j],2) + pow(points_y[i][j],2))
            points_x[i][j] = points_x[i][j]/(1 + k1 * pow(r,2) + k2 * pow(r,4))
            points_y[i][j] = points_y[i][j]/(1 + k1 * pow(r,2) + k2 * pow(r,4))

    # print(points_x)

    # remove outlier
    mask = (points_z > 0) & (points_z < 0.60) & (points_x < 0.2) & (points_x < 0.2) & (points_y < 0.2) & (points_y < 0.2)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)
    # print(points.min(axis=0), points.max(axis=0))

    # get prediction
    gg, cloud = anygrasp.get_grasp(points, colors, lims)

    if len(gg) == 0:
        print('No Grasp detected after collision detection!')
        gg_found = False
    else:
        # gg = angle_control(gg,20)
        gg = gg.nms().sort_by_score()
        gg_pick = gg[0:20]
        gg_found = gg_pick[0]
    
    # visualization
    if cfgs.debug:
        trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        cloud.transform(trans_mat)
        grippers = gg.to_open3d_geometry_list()
        for gripper in grippers:
            gripper.transform(trans_mat)
        n_points = 80000
        points_array = np.asarray(cloud.points)

        # 获取点云数据的维度
        # point_cloud_dimensions = points_array.shape
        # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin = [0, 0, 0])
        o3d.visualization.draw_geometries([*grippers, cloud])
        # print(point_cloud_dimensions)
        # o3d.visualization.draw_geometries([cloud.uniform_down_sample(n_points)])
        # o3d.visualization.draw_geometries([cloud])
        # o3d.io.write_point_cloud("/home/ani/Desktop/example_data/downsampled_point_cloud.ply", cloud)

    return gg_found

def requestsrv(gg):
    rpy_pose_client = rospy.ServiceProxy('rpypose',rpypose)
    rpy_pose_client.wait_for_service()
    request = rpyposeRequest()
    request.x = gg.translation[0]
    request.y = gg.translation[1]
    request.z = gg.translation[2]
    # print(tf.transformations.euler_from_matrix(gg.rotation_matrix))
    x_rotate = np.mat([[1,0,0],[0,0,-1],[0,1,0]])
    z_rotate = np.mat([[0,-1,0],[1,0,0],[0,0,1]])
    gg.rotation_matrix = np.dot(gg.rotation_matrix,z_rotate)
    gg.rotation_matrix = np.dot(gg.rotation_matrix,x_rotate)
    (request.roll,request.pitch,request.yaw) = tf.transformations.euler_from_matrix(gg.rotation_matrix)
    # print("( %.5f , %.5f, %.5f )"%(request.roll,request.pitch,request.yaw))
    if request.yaw < 0:
         z_rotate_pi = np.mat([[-1,0,0],[0,-1,0],[0,0,1]])
         gg.rotation_matrix = np.dot(gg.rotation_matrix,z_rotate_pi)
         (request.roll,request.pitch,request.yaw) =  tf.transformations.euler_from_matrix(gg.rotation_matrix)
        #  print("( %.5f , %.5f, %.5f )"%(request.roll,request.pitch,request.yaw))
    request.width = gg.width
    request.home = False
    rep = rpy_pose_client.call(request)


if __name__ == '__main__':

    # rospy.init_node('kinect_subscriber')
    gg_target = kinect_subscriber()
    # while(gg_target):
    #      print("detected!")
    #      requestsrv(gg_target)
    #      gg_target = kinect_subscriber()




