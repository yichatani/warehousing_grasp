#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, sys, rospkg
sys.path.insert(0, rospkg.RosPack().get_path("Anygrasp_sdk"))

from Anygrasp_sdk.srv import rpypose, rpyposeRequest, rpyposeResponse
from moveit_interface import *

class Planner:
    def __init__(self):        
        self.moveit_interface = MoveItInterface()
        self.pose_server = rospy.Service('rpypose', rpypose, self.rpypose_cb)
        
    def rpypose_cb(self, rpypose):
        # if rpypose.home:
        #     self.moveit_interface.go_home()
        # else:   
        #     self.moveit_interface.move_p([rpypose.x, rpypose.y, rpypose.z, rpypose.roll, rpypose.pitch, rpypose.yaw])
        self.moveit_interface.log_pos()


        return rpyposeResponse()
    
if __name__ == '__main__':
    rospy.init_node('planner', anonymous=True)
    planner = Planner()
    rospy.spin()