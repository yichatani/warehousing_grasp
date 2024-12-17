#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge,CvBridgeError
import message_filters
import open3d as o3d
import numpy as np

bridge = CvBridge()

def rgb_callback(data):
    
    cv_rgb = bridge.imgmsg_to_cv2(data, "bgr8") 
    # cv2.imwrite("/home/ani/Desktop/example_data/" + str(rospy.Time.now().to_sec()) + "color.png",cv_rgb)
    global index 
    index = index + 1
    cv2.imwrite("/home/ani/Desktop/example_data/tracking/data2/color_%03d.png"%index,cv_rgb)
    
    # kinect_subscriber1().color.unregister() # 节点只订阅一次

def depth_callback(data):

    cv_depth = bridge.imgmsg_to_cv2(data, "16UC1") 
    # cv2.imwrite("/home/ani/Desktop/example_data/" + str(rospy.Time.now().to_sec()) + "depth.png",cv_depth)
    cv2.imwrite("/home/ani/Desktop/example_data/tracking/data2/depth_%03d.png"%index,cv_depth)
    # kinect_subscriber1().depth.unregister() # 节点只订阅一次

def kinect_subscriber1():
    
    color = rospy.Subscriber('/rgb/image_raw', Image,rgb_callback)
    depth = rospy.Subscriber('/depth_to_rgb/image_raw',Image,depth_callback)
    rospy.spin()

#------------------------------------------------------------------------------
index = 0
def color_depth_callback(data1, data2):
    global index
    cv_rgb = bridge.imgmsg_to_cv2(data1, "bgr8")
    cv_rgb = cv2.cvtColor(cv_rgb,cv2.COLOR_BGR2RGB)
    cv_depth = bridge.imgmsg_to_cv2(data2, "16UC1")
    cv_rgb = o3d.geometry.Image(cv_rgb)
    cv_depth = o3d.geometry.Image(cv_depth)
    fx, fy = 969.750743, 967.848535
    cx, cy = 1031.790933, 771.076245
    width , height = 2048, 1536

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    cv_rgb, cv_depth, depth_scale=1000.0, depth_trunc = 3.0, convert_rgb_to_intensity=False)

    intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    # cv2.imwrite("/home/ani/Desktop/dynamic/ob4/color_%03d.png"%index,cv_rgb)
    # cv2.imwrite("/home/ani/Desktop/dynamic/ob4/depth_%03d.png"%index,cv_depth)
    o3d.visualization.draw_geometries([cloud])
    o3d.io.write_point_cloud("/home/ani/Desktop/example_data/cloud_point/point_cloud_%03d.ply"%index, cloud)
    index = index + 1

def kinect_subscriber():
    color = message_filters.Subscriber('/rgb/image_raw', Image)
    depth = message_filters.Subscriber('/depth_to_rgb/image_raw',Image)
    color_depth = message_filters.ApproximateTimeSynchronizer([color, depth],100,0.1,allow_headerless=True)
    color_depth.registerCallback(color_depth_callback)
    
    rospy.spin()
#-----------------------------------------------------------------------------------
# def color_depth_callback(data1, data2):

#     cv_rgb = bridge.imgmsg_to_cv2(data1, "bgr8")
#     cv_depth = bridge.imgmsg_to_cv2(data2, "16UC1") 
    

# def kinect_subscriber():
#     color = message_filters.Subscriber('/rgb/image_raw', Image)
#     depth = message_filters.Subscriber('/depth_to_rgb/image_raw',Image)
#     color_depth = message_filters.ApproximateTimeSynchronizer([color, depth],100,0.0001,allow_headerless=True)
#     color_depth.registerCallback(color_depth_callback)
    
#     rospy.spin()

#-----------------------------------------------------------------------------------
# def kinect_subscriber():
#     index = 0
#     while not rospy.is_shutdown():
#         color = rospy.wait_for_message('/rgb/image_raw',Image,timeout=None)
#         depth = rospy.wait_for_message('/depth_to_rgb/image_raw',Image,timeout=None)
#         cv_rgb = bridge.imgmsg_to_cv2(color, "bgr8")
#         cv2.imwrite("/home/ani/Desktop/example_data/tracking/color_%03d.png"%index,cv_rgb)
#         cv_depth = bridge.imgmsg_to_cv2(depth, "16UC1") 
#         cv2.imwrite("/home/ani/Desktop/example_data/tracking/depth_%03d.png"%index,cv_depth)
#         index = index + 1


if __name__ == '__main__':

    rospy.init_node('kinect_subscriber1')

    kinect_subscriber()