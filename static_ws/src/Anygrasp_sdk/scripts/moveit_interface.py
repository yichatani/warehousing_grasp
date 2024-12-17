#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, rospkg, sys, math, yaml, std_msgs
import moveit_commander
from moveit_commander import MoveGroupCommander, PlanningSceneInterface
from moveit_msgs.msg import  PlanningScene, ObjectColor
from moveit_msgs.srv import ApplyPlanningScene
from geometry_msgs.msg import Pose, PoseStamped
from tf.transformations import quaternion_from_euler

pi = math.pi

class MoveItInterface:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)

        self.base_frame = 'base_link'

        self.group = MoveGroupCommander("manipulator")
        self.group.set_max_velocity_scaling_factor(1)
        self.group.set_max_acceleration_scaling_factor(1)
        self.group.set_goal_position_tolerance(0.001)
        self.group.set_goal_joint_tolerance(0.001)
        self.group.set_goal_orientation_tolerance(0.01)
        self.group.set_planning_time(5)
        self.group.set_num_planning_attempts(5)
        self.group.set_planner_id("RRTConnect")
        self.group.set_pose_reference_frame(self.base_frame)
        self.group.allow_replanning(True)

        self.scene = PlanningSceneInterface()
        self.scene_pub = rospy.Publisher('planning_scene', PlanningScene, queue_size=5)
        self.colors = dict()

        self.set_scene()
        # self.go_home()
        self.log_pos()

    def __del__(self):
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)
    
    def go_home(self, a = 1, v = 1):
        self.set_state(a, v)
        self.group.set_named_target('up')
        self.group.go()
        rospy.sleep(1)

    def move_j(self, joint_goal = [0, -pi/2, 0, -pi/2, 0, 0], a = 1, v = 1):
        self.set_state(a, v)
        self.group.set_joint_value_target(joint_goal)
        rospy.loginfo("\n\nmove_j:" + str(joint_goal) + "\n\n")
        self.group.go()
        rospy.sleep(1)
    
    def move_p(self, tool_goal = [0.3, 0, 0.3, 0, -pi/2, 0], a = 1, v = 1):
        self.set_state(a, v)
        self.group.set_pose_target(tool_goal)
        rospy.loginfo("\n\nmove_p:" + str(tool_goal) + "\n\n")
        self.group.go()
        rospy.sleep(1)
    
    def move_l(self, tool_goals = [[0.3, 0, 0.3, 0, -pi/2, 0]], a = 1, v = 1):
        
        waypoints = []
        for tool_goal in tool_goals: 
            pt = Pose()
            q = quaternion_from_euler(*tool_goal[3:6])
            pt.position.x, pt.position.y, pt.position.z = tool_goal[0], tool_goal[1], tool_goal[2]
            pt.orientation.x, pt.orientation.y, pt.orientation.z, pt.orientation.w = q[0], q[1], q[2], q[3]
            waypoints.append(pt)
        rospy.loginfo("\n\nmove_l:" + str(tool_goals) + "\n\n")
        
        self.set_state(a, v)

        fraction, maxtries, attempts = 0.0, 100, 0
        while fraction < 1.0 and attempts < maxtries:
            (plan, fraction)  = self.group.compute_cartesian_path(waypoints, eef_step = 0.01, jump_threshold = 0.0)
            attempts += 1

        if fraction == 1.0:
            rospy.loginfo("Move_l computed successfully. Moving the arm.")
            self.group.execute(self.scale_trajectory_speed(plan, 0.3))
            rospy.sleep(1)
        else:
            rospy.loginfo("Path planning failed with only {0:.4f} success after {1} attempts.".format(fraction, maxtries))
        
        rospy.sleep(1)

    def set_state(self, a = 1, v = 1):
        self.group.set_max_acceleration_scaling_factor(a)
        self.group.set_max_velocity_scaling_factor(v)
        self.group.set_start_state_to_current_state()       

    def scale_trajectory_speed(self, plan, scale):
        n_joints = len(plan.joint_trajectory.joint_names)
        n_points = len(plan.joint_trajectory.points)
        print(n_joints)
        print(n_points)
        for i in range(n_points):
            plan.joint_trajectory.points[i].time_from_start *= 1/scale
            plan.joint_trajectory.points[i].velocities = [v * scale for v in plan.joint_trajectory.points[i].velocities]
            plan.joint_trajectory.points[i].accelerations = [a * scale * scale for a in plan.joint_trajectory.points[i].accelerations]
        return plan

    def set_scene(self):
        rospy.sleep(1)
        self.scene.remove_world_object("ground")
        rospy.sleep(1)

        hr = std_msgs.msg.Header()
        hr.frame_id = self.base_frame

        # with open(rospkg.RosPack().get_path("ur_planner") + "/config/scene.yaml", 'r') as file:
        #     data = yaml.safe_load(file)
        #     for i in range(int(data['num'])):
        #         name = data['obj'+str(i)]['name']
        #         size, color, pose = data['obj'+str(i)]['size'], data['obj'+str(i)]['color'], data['obj'+str(i)]['pose']
        #         pt_s = PoseStamped()
        #         pt_s.header = hr
        #         pt_s.pose.position.x = pose[0]
        #         pt_s.pose.position.y = pose[1]
        #         pt_s.pose.position.z = pose[2]
        #         pt_s.pose.orientation.w = 1.0
        #         self.scene.add_box(name, pt_s, size)
        #         self.set_color(name, color)            

    def set_color(self, name, color = (0, 0, 0, 0.9)):
        color = ObjectColor(id = name, color = std_msgs.msg.ColorRGBA(*color))
        self.colors[name] = color

        p = PlanningScene()
        p.is_diff = True
        for color in self.colors.values():
            p.object_colors.append(color)
        self.scene_pub.publish(p)
    
    def log_pos(self):
        rospy.loginfo_throttle(1, self.cur_p())
        rospy.loginfo_throttle(1, [round(j, 3) for j in self.cur_j()])
        rospy.loginfo_throttle(1, [round(theta, 3) for theta in self.group.get_current_rpy()])

    def cur_p(self):
        return self.group.get_current_pose().pose
    
    def cur_j(self):
        return self.group.get_current_joint_values()

if __name__ == '__main__':
    try:
        rospy.init_node('planner', anonymous=True)
        moveit_control = MoveItInterface()
        moveit_control.go_home()
        moveit_control.move_j([0, -pi/2, 0, -pi/2, 0, 0])
        moveit_control.move_p([0.3, 0, 0.3, 0, -pi/2, 0])
        moveit_control.move_l([[0.3, 0, 0.3, 0, -pi/2, 0], [0.3, 0.2, 0.3, 0, -pi/2, 0]])
        moveit_control.move_j([0, -pi/2, 0, -pi/2, 0, 0])
        moveit_control.go_home()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass