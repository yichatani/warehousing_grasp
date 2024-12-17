#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import sys
import tf
import rospy
import argparse
import numpy as np
import open3d as o3d
from cv_bridge import CvBridge,CvBridgeError
import message_filters
from PIL import Image as Ig
from sensor_msgs.msg import Image
from graspnetAPI import GraspGroup
from demo_planner import MoveIt_Control
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

# 全局变量区
index = 0
grasp_ids = [0] # 这是跳出循环后需要一并修改回到[0]的值
vis = o3d.visualization.Visualizer()
anygrasp_tracker = AnyGraspTracker(cfgs)
anygrasp_tracker.load_net()
moveit_server = MoveIt_Control()


# 用以存储书架格子位置的末端位姿
#-------------------------------------------------------
# # 创建一个向量
# vector_np = np.array([3, 4])

# # 创建多个向量
# vectors_np = np.array([[3, 4], [1, 2], [5, 6]])

# # 访问和修改与普通Python列表类似
# print(vectors_np[0])  # 访问第一个向量
# vectors_np[2][1] = 9  # 修改第三个向量的y分量
#--------------------------------------------------------


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
    for g in gg:
        g_x_vector = g.rotation_matrix[:, 0]
        z_vector = np.array([0, 0, 1])
        if angle_between_vectors(g_x_vector,z_vector) <=angle:
            ggs.add(g)
        # print(angle_between_vectors(g_x_vector,z_vector))
    # gg = ggs
    return ggs

# 生成目标夹爪位姿
def generate_grasp(points,colors):
    global index
    global grasp_ids
    global anygrasp_tracker
    target_gg, curr_gg, target_grasp_ids, corres_preds = anygrasp_tracker.update(points, colors, grasp_ids)
    if index == 0:
        # select grasps on objects to track for the 1st frame
        grasp_mask_x = ((curr_gg.translations[:,0]>-0.18) & (curr_gg.translations[:,0]<0.18))
        grasp_mask_y = ((curr_gg.translations[:,1]>-0.12) & (curr_gg.translations[:,1]<0.12))
        grasp_mask_z = ((curr_gg.translations[:,2]>0.20) & (curr_gg.translations[:,2]<0.55))
        grasp_ids = np.where(grasp_mask_x & grasp_mask_y & grasp_mask_z)[0][::]   
        target_gg = curr_gg[grasp_ids]
    else:
        grasp_ids = target_grasp_ids
        # print(i, target_grasp_ids)
    target_gg = angle_control(target_gg,15)
    target_gg = target_gg.nms().sort_by_score()
    return target_gg

# 可视化
def visual(target_gg,points,colors,camera):
    global vis
    if index == 0:
        vis.create_window(height = camera.height, width = camera.width)
    trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    cloud.transform(trans_mat)
    grippers = target_gg.to_open3d_geometry_list()
    for gripper in grippers:
        gripper.transform(trans_mat)
    vis.add_geometry(cloud)
    for gripper in grippers:
        vis.add_geometry(gripper)
    vis.poll_events()
    vis.remove_geometry(cloud)
    for gripper in grippers:
        vis.remove_geometry(gripper)

# 将rgb和depth存到某一文件夹中
def write_file(rgb,depth,dir):
    global index
    cv2.imwrite(dir + "color_%03d.png"%index,rgb)
    cv2.imwrite(dir + "depth_%03d.png"%index,depth)
    # dir = "/home/ani/Desktop/example_data/tracking/data5/"

def path_planning(target,end_cur):
    global index
    moveit_server.move_p([end_cur[0],end_cur[1],target['z'],end_cur[3],end_cur[4],end_cur[5]])
    moveit_server.gripper(target['width']-0.01)
    # 固定的路径规划
    moveit_server.move_l()# 先到某个位置
    moveit_server.move_l()# 再运动到架子的某一格上
    moveit_server.gripper(0.085)# 打开夹爪
    moveit_server.move_l()# 运动回某位置
    moveit_server.move_j()# 运动至起始位置
    index = 0

# 位姿转换模块
def tf_transform(target_gg):
    request = rpyposeRequest()
    request.x = target_gg.translations[0,0]
    request.y = target_gg.translations[0,1]
    request.z = target_gg.translations[0,2]
    # print(tf.transformations.euler_from_matrix(target_gg.rotation_matrix))
    x_rotate = np.mat([[1,0,0],[0,0,-1],[0,1,0]])
    z_rotate = np.mat([[0,-1,0],[1,0,0],[0,0,1]])
    target_gg.rotation_matrices[0] = np.dot(target_gg.rotation_matrices[0],z_rotate)
    target_gg.rotation_matrices[0] = np.dot(target_gg.rotation_matrices[0],x_rotate)
    (request.roll,request.pitch,request.yaw) = tf.transformations.euler_from_matrix(target_gg.rotation_matrices[0])
    # print("( %.5f , %.5f, %.5f )"%(request.roll,request.pitch,request.yaw))
    if request.yaw < 0:
        z_rotate_pi = np.mat([[-1,0,0],[0,-1,0],[0,0,1]])
        target_gg.rotation_matrices[0] = np.dot(target_gg.rotation_matrices[0],z_rotate_pi)
        (request.roll,request.pitch,request.yaw) =  tf.transformations.euler_from_matrix(target_gg.rotation_matrices[0])
        # print("( %.5f , %.5f, %.5f )"%(request.roll,request.pitch,request.yaw))
    request.width = target_gg.widths[0]
    request.home = False

    target = moveit_server.tf_eye_base(request)
    return target

def launch_grasp(target,end_cur,dist):
    if dist <= 0.01:
        path_planning(target,end_cur)
    else:
        # 控制机械臂只能进行平面抓取和移动，roll、pitch都被锁住，只允许其进行x、y的平移和yaw的旋转，再加上执行抓取时z方向的移动。
        moveit_server.move_p([target["x"], target["y"], end_cur[2], end_cur[3], end_cur[4], target["yaw"]],a=0.5,v=0.5)

# tracker的回调函数
def color_depth_callback(data1, data2):
    try:
        # 初始化全局变量
        global vis
        global index
        cv_rgb = bridge.imgmsg_to_cv2(data1, "bgr8")
        cv_rgb = cv2.cvtColor(cv_rgb,cv2.COLOR_BGR2RGB)
        cv_depth = bridge.imgmsg_to_cv2(data2, "32FC1")
        # write_file(cv_rgb,cv_depth,'/home/ani/Desktop/example_data/tracking/data7/') 
        # cv_rgb2 = Ig.open(os.path.join("/home/ani/Desktop/example_data/tracking/data4", 'color_%03d.png'%index))
        colors = np.array(cv_rgb, dtype=np.float32) / 255.0
        depths = np.array(cv_depth, dtype=np.float32)
        # set camera intrinsics
        width, height = depths.shape[1], depths.shape[0]
        fx, fy = 969.750743, 967.848535
        cx, cy = 1031.790933, 771.076245
        scale = 1000.0
        camera = CameraInfo(width, height, fx, fy, cx, cy, scale)
            
        points = create_point_cloud_from_depth_image(depths, camera)
        mask = (points[:,:,2] > 0) & (points[:,:,2] < 1.0)
        points = points[mask].astype(np.float32)
        colors = colors[mask].astype(np.float32)
        
        # 抓取生成模块
        target_gg = generate_grasp(points,colors)
        if len(target_gg) == 0:
            moveit_server.move_j()# 回到初始位置
            index = 0
        else:
            # 位姿转换模块
            target = tf_transform(target_gg)
            end_cur = moveit_server.cur_p
            print(target)
            print(end_cur)

            # 计算当前位姿和目标位姿水平方向上的欧几里得距离
            target_vec2 = np.array([target['x'], target['y']])
            target_vec3 = np.array([target['x'], target['y'], target['z']])
            end_cur_vec2 = np.array([end_cur[0], end_cur[1]])
            end_cur_vec3 = np.array([end_cur[0], end_cur[1]], end_cur[2])
            dist2 = np.linalg.norm(target_vec2 - end_cur_vec2)
            dist3 = np.linalg.norm(target_vec3 - end_cur_vec3)
            if dist3 < 0.055: # 若距离够近，追踪
                launch_grasp(target,end_cur,dist2)
                index += 1
            else: # 反之，放弃追踪回到原位
                moveit_server.move_j()# 回到初始位置
                index = 0
        
        # 可视化模块：
        # visual(target_gg,points,colors,camera)
        
    except Exception as e:
        rospy.logerr("处理消息时发生错误: %s" % str(e))


# 目前用的是用ros驱动的kinect同时订阅rgb图和深度图
def tracker():
    color = message_filters.Subscriber('/rgb/image_raw', Image)
    depth = message_filters.Subscriber('/depth_to_rgb/image_raw',Image)
    color_depth = message_filters.ApproximateTimeSynchronizer([color, depth],100,0.5,allow_headerless=True)
    color_depth.registerCallback(color_depth_callback)
    rospy.spin()
    


if __name__ == "__main__":
    rospy.init_node('kinect_subscriber')
    # moveit_server.move_p([0.486331991094792, 0.63709065003677, 0.4975897409274045, 3.041780318015747, 0.05669433384859317, 1.1912188864171005],a=1,v=1)
    tracker()
