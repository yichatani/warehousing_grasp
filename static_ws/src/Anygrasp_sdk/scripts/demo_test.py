#! /usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image
import cv2
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
import matplotlib.pyplot as plt

flag = 0
index = 0

def color_depth_callback(data1,data2):
    global flag
    global index
    print(index)
    index += 1
    if index == 20:
        flag = 1


# 目前用的是用ros驱动的kinect同时订阅rgb图和深度图
def tracker():
    color = message_filters.Subscriber('/rgb/image_raw', Image)
    depth = message_filters.Subscriber('/depth_to_rgb/image_raw',Image)
    color_depth = message_filters.ApproximateTimeSynchronizer([color, depth],100,0.1)
    color_depth.registerCallback(color_depth_callback)
    rospy.spin()
    


if __name__ == "__main__":
    rospy.init_node('kinect_subscriber')
    # moveit_server.move_p([0.486331991094792, 0.63709065003677, 0.4975897409274045, 3.041780318015747, 0.05669433384859317, 1.1912188864171005],a=1,v=1)
    tracker()
    