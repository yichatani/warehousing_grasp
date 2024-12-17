#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg

def gripper(width):
    # 初始化ROS节点
    rospy.init_node("Robotiq_close_open")
    # 创建发布者对象
    pub = rospy.Publisher(
        "Robotiq2FGripperRobotOutput", outputMsg.Robotiq2FGripper_robot_output, queue_size=1
    )
    command = outputMsg.Robotiq2FGripper_robot_output()
    command.rACT = 1
    command.rGTO = 1
    command.rSP = 255
    command.rFR = 150

    max_width = 0.085
    command.rPR = int(255 * (1-width/max_width))

    for i in range(3):
        pub.publish(command)
        rospy.sleep(0.1)

if __name__ == "__main__":
    gripper(0.085)#------max open
    rospy.sleep(1.5)
    gripper(0)#------close


