#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os,sys
import argparse
import numpy as np
import open3d as o3d
from PIL import Image
from graspnetAPI import GraspGroup
from Anygrasp_sdk.srv import rpypose, rpyposeRequest, rpyposeResponse
from demo_planner import *
from tracker import AnyGraspTracker

# path = os.path.abspath(".")
# sys.path.insert(0,path + "/grasp_tracking/log")

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=False,default='/home/ani/anygrasp_sdk/grasp_tracking/log/checkpoint_tracking.tar', help='Model checkpoint path')
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
        print(angle_between_vectors(g_x_vector,z_vector))
    return ggss

def get_data(data_dir, index):
    # load image
    cv_rgb = Image.open(os.path.join(data_dir, 'color_%03d.png'%index))
    cv_depth = Image.open(os.path.join(data_dir, 'depth_%03d.png'%index))
    # cv_depth = np.load(os.path.join(data_dir, 'depth_%03d.npy'%index))

    # colors = np.array(Image.open(os.path.join(data_dir, 'color_%03d.png'%index)), dtype=np.float32) / 255.0
    # depths = np.load(os.path.join(data_dir, 'depth_%03d.npy'%index))

    colors = np.array(cv_rgb, dtype=np.float32) / 255.0
    depths = np.array(cv_depth,dtype=np.float32)

    # set camera intrinsics
    width, height = depths.shape[1], depths.shape[0]
    # fx, fy = 969.750743, 967.848535
    # cx, cy = 1031.790933, 771.076245
    fx, fy = 927.17, 927.37
    cx, cy = 651.32, 349.62
    scale = 1000.0
    camera = CameraInfo(width, height, fx, fy, cx, cy, scale)

    # get point cloud
    points = create_point_cloud_from_depth_image(depths, camera)
    mask = (points[:,:,2] > 0.1) & (points[:,:,2] < 0.5) & (points[:,:,0] < 0.2) & (points[:,:,0] < 0.2) & (points[:,:,1] < 0.2) & (points[:,:,1] < 0.2)
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)

    return points, colors

def demo(data_dir_list, indices):
    # intialization
    anygrasp_tracker = AnyGraspTracker(cfgs)
    anygrasp_tracker.load_net()

    vis = o3d.visualization.Visualizer()
    # vis.create_window(height=720, width=1280)
    vis.create_window(height=1536 , width=2048 )

    grasp_ids = [0]
    for i in range(len(indices)):
        # get prediction
        points, colors = get_data(data_dir_list, indices[i])
        target_gg, curr_gg, target_grasp_ids, corres_preds = anygrasp_tracker.update(points, colors, grasp_ids)
        if i == 0:
            # select grasps on objects to track for the 1st frame
            grasp_mask_x = ((curr_gg.translations[:,0]>-0.10) & (curr_gg.translations[:,0]<0.10))
            grasp_mask_y = ((curr_gg.translations[:,1]>-0.12) & (curr_gg.translations[:,1]<0.12))
            grasp_mask_z = ((curr_gg.translations[:,2]>0.05) & (curr_gg.translations[:,2]<0.6))
            grasp_ids = np.where(grasp_mask_x & grasp_mask_y & grasp_mask_z)[0][::]
            target_gg = curr_gg[grasp_ids]
        else:
            grasp_ids = target_grasp_ids
        # print(i, target_grasp_ids)
        if len(target_gg) == 0:
            print("0000000000000000000000000")
        target_gg = angle_control(target_gg,30)
        target_gg = target_gg.nms().sort_by_score()
        # visualization
        if cfgs.debug:
            trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            cloud.colors = o3d.utility.Vector3dVector(colors)
            cloud.transform(trans_mat)
            grippers = target_gg.to_open3d_geometry_list()
            # for gripper in grippers:
            #     gripper.transform(trans_mat)
            grippers[0].transform(trans_mat)
            vis.add_geometry(cloud)
            # for gripper in grippers:
            #     vis.add_geometry(gripper)
            vis.add_geometry(grippers[0])
            vis.poll_events()
            vis.remove_geometry(cloud)
            # for gripper in grippers:
            #     vis.remove_geometry(gripper)
            vis.remove_geometry(grippers[0])


        # 机械臂运动模块
        
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
        # request.width = target_gg.widths
        # request.home = False

        # target = moveit_server.tf_eye_base(request)
        # moveit_server.move_l([target["x"], target["y"], target["z"], target["roll"], target["pitch"], target["yaw"]],a=1,v=1)

        # moveit_server.move_l([0.6, 0, 0.5, np.pi, 0, np.pi],a=1,v=1)
        # # print(moveit_server.cur_p())
        # print(moveit_server.cur_j())


if __name__ == "__main__":
    # data_dir = "/home/ani/anygrasp_sdk/grasp_tracking/example_data/original_data"
    # data_dir_list = [x for x in range(30)]
    # rospy.init_node('tracking')
    # moveit_server = MoveIt_Control()
    # moveit_server.prepare()
    # data_dir = "/home/ani/Desktop/example_data/tracking/data4"
    # data_dir = "/home/ani/static_ws/example_data"
    data_dir = "/home/ani/Desktop/dynamic/ob3"
    data_dir_list = [x for x in range(53,58)]

    demo(data_dir, data_dir_list)