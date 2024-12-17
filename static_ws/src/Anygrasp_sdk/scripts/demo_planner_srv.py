#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy, sys, rospkg
sys.path.insert(0, rospkg.RosPack().get_path("Anygrasp_sdk"))

from Anygrasp_sdk.srv import rpypose, rpyposeRequest, rpyposeResponse
from demo_planner import MoveIt_Control

class Planner:
    def __init__(self):        
        self.moveit_control = MoveIt_Control()
        self.pose_server = rospy.Service('rpypose', rpypose, self.rpypose_cb)
        self.moveit_control.prepare()
        
    def rpypose_cb(self, rpypose):
        # self.moveit_control.install()
        self.moveit_control.prepare()
        rospy.sleep(0.1)
        # print(self.moveit_control.cur_p())
        print(self.moveit_control.cur_j())
        self.static_grasp(rpypose)
        return rpyposeResponse()
    
    def static_grasp(self,rpypose):
        target = self.moveit_control.tf_eye_base(rpypose)
        # print(target)
        self.moveit_control.move_l([target["x"], target["y"], target["z"]+0.1, target["roll"], target["pitch"], target["yaw"]],a=2,v=2)
        self.moveit_control.move_l([target["x"], target["y"], target["z"], target["roll"], target["pitch"], target["yaw"]],a=2,v=2)
        self.moveit_control.gripper(target["width"]-0.008)# 抓取
        rospy.sleep(1)
        self.moveit_control.move_l([target["x"], target["y"], target["z"]+0.1, target["roll"], target["pitch"], target["yaw"]],a=2,v=2)
        # # self.moveit_control.move_l([0.65, 0.4, 0.5, np.pi, 0, np.pi],a=5,v=5)# 放置点（手动设置）
        # self.moveit_control.gripper(0.085)# 打开夹爪
        # rospy.sleep(1)
        # self.moveit_control.prepare()# 回到准备位置


if __name__ == '__main__':
    rospy.init_node('planner', anonymous=True)
    planner = Planner()
    rospy.spin()