#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import rospy
import argparse
import numpy as np
import open3d as o3d
from PIL import Image as Ig
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from graspnetAPI import GraspGroup
from demo_planner import *
from Anygrasp_sdk.srv import rpypose, rpyposeRequest, rpyposeResponse

bridge = CvBridge()

path = os.path.abspath(".")
sys.path.insert(0,path + "/src/Anygrasp_sdk/scripts")

from tracker import AnyGraspTracker

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=False, default='log/checkpoint_tracking.tar', help='Model checkpoint path')
parser.add_argument('--filter', type=str, default='oneeuro', help='Filter to smooth grasp parameters(rotation, width, depth). [oneeuro/kalman/none]')
parser.add_argument('--debug', action='store_true',default=True, help='Enable visualization')
parser.add_argument('--top_down_grasp', action='store_true', default=True, help='Output top-down grasps')
parser.add_argument('--max_gripper_width', type=float, default=0.085, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.023, help='Gripper height')
cfgs = parser.parse_args()

class CameraInfo:
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

def create_point_cloud_from_depth_image(depth, camera, organized=True):
    assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    points = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        points = points.reshape([-1, 3])
    return points


def get_data():
    # load image
    # colors = np.array(Image.open(os.path.join(data_dir, 'color_%03d.png'%index)), dtype=np.float32) / 255.0
    # depths = np.load(os.path.join(data_dir, 'depth_%03d.npy'%index))

    colors = rospy.wait_for_message('/rgb/image_raw',Image,timeout=None)
    depths = rospy.wait_for_message('/depth_to_rgb/image_raw',Image,timeout=None)
    cv_rgb = bridge.imgmsg_to_cv2(colors, "bgr8")
    cv_depth = bridge.imgmsg_to_cv2(depths, "16UC1")
     
    colors = np.array(cv_rgb, dtype=np.float32) / 255.0
    depths = np.array(cv_depth)

    # set camera intrinsics
    width, height = depths.shape[1], depths.shape[0]
    fx, fy = 969.750743, 967.848535
    cx, cy = 1031.790933, 771.076245
    scale = 1000.0
    camera = CameraInfo(width, height, fx, fy, cx, cy, scale)

    # get point cloud
    points = create_point_cloud_from_depth_image(depths, camera)
    mask = (points[:,:,2] > 0) & (points[:,:,2] < 1.5)
    points = points[mask]
    colors = colors[mask]

    return points, colors

def tracker():
    # intialization
    anygrasp_tracker = AnyGraspTracker(cfgs)
    anygrasp_tracker.load_net()
    
    vis = o3d.visualization.Visualizer()
    # vis.create_window(height=720, width=1280)
    vis.create_window(height=1536, width=2048)

    grasp_ids = [0]
    i = 0
    while not rospy.is_shutdown():
        # get prediction
        points, colors = get_data()
        target_gg, curr_gg, target_grasp_ids, corres_preds = anygrasp_tracker.update(points, colors, [grasp_ids[0]])

        if i == 0:
            # select grasps on objects to track for the 1st frame
            grasp_mask_x = ((curr_gg.translations[:,0]>-0.20) & (curr_gg.translations[:,0]<0.20))
            grasp_mask_y = ((curr_gg.translations[:,1]>-0.20) & (curr_gg.translations[:,1]<0.20))
            grasp_mask_z = ((curr_gg.translations[:,2]>0.10) & (curr_gg.translations[:,2]<0.50))
            grasp_ids = np.where(grasp_mask_x & grasp_mask_y & grasp_mask_z)[0][::]
            # print(grasp_ids)
            for j in range(len(grasp_ids)): # 选取第一帧置信度最高的抓取进行tracking，所以将grasp_ids按照抓取分数进行排序，冒泡排序
                for k in range(0, len(grasp_ids)-j-1):
                    if curr_gg[[grasp_ids[k]]].scores < curr_gg[[grasp_ids[k+1]]].scores:
                        grasp_ids[k],grasp_ids[k+1] = grasp_ids[k+1],grasp_ids[k]
            i = 1
            target_gg = curr_gg[[grasp_ids[0]]]
        else:
            grasp_ids = target_grasp_ids
        # print(i, target_grasp_ids)
        # print(i, grasp_ids)
        # print(i, corres_preds[1])

        # visualization
        if cfgs.debug:
            trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            cloud.colors = o3d.utility.Vector3dVector(colors)
            cloud.transform(trans_mat)
            grippers = target_gg.to_open3d_geometry_list()
            
            curr_grippers = curr_gg.to_open3d_geometry_list()

            for gripper in grippers:
                gripper.transform(trans_mat)
            vis.add_geometry(cloud)
            for gripper in grippers:
                vis.add_geometry(gripper)
            vis.poll_events()
            vis.remove_geometry(cloud)
            for gripper in grippers:
                vis.remove_geometry(gripper)

        # # 机械臂运动模块
        # request = rpyposeRequest()
        # request.x = target_gg.translations[0,0]
        # request.y = target_gg.translations[0,1]
        # request.z = target_gg.translations[0,2]
        # # print(tf.transformations.euler_from_matrix(target_gg.rotation_matrix))
        # x_rotate = np.mat([[1,0,0],[0,0,-1],[0,1,0]])
        # z_rotate = np.mat([[0,-1,0],[1,0,0],[0,0,1]])
        # target_gg.rotation_matrices[0] = np.dot(target_gg.rotation_matrices[0],z_rotate)
        # target_gg.rotation_matrices[0] = np.dot(target_gg.rotation_matrices[0],x_rotate)
        # (request.roll,request.pitch,request.yaw) = tf.transformations.euler_from_matrix(target_gg.rotation_matrices[0])
        # print("( %.5f , %.5f, %.5f )"%(request.roll,request.pitch,request.yaw))
        # if request.yaw < 0:
        #     z_rotate_pi = np.mat([[-1,0,0],[0,-1,0],[0,0,1]])
        #     target_gg.rotation_matrices[0] = np.dot(target_gg.rotation_matrices[0],z_rotate_pi)
        #     (request.roll,request.pitch,request.yaw) =  tf.transformations.euler_from_matrix(target_gg.rotation_matrices[0])
        #     print("( %.5f , %.5f, %.5f )"%(request.roll,request.pitch,request.yaw))
        # request.width = target_gg.widths[0]
        # request.home = False

        # target = moveit_server.tf_eye_base(request)
        # moveit_server.move_p([target["x"], target["y"], 0.4, target["roll"], target["pitch"], np.pi],a=1,v=1)



if __name__ == "__main__":
    rospy.init_node('kinect_subscriber')
    # moveit_server = MoveIt_Control()
    # moveit_server.prepare_tracking()
    tracker()