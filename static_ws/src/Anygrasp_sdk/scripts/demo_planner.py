#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 导入基本ros和moveit库
import rospy, sys, math
import moveit_commander
from moveit_commander import MoveGroupCommander, PlanningSceneInterface, RobotCommander
from moveit_msgs.msg import  PlanningScene, ObjectColor,CollisionObject, AttachedCollisionObject,Constraints,OrientationConstraint
from geometry_msgs.msg import PoseStamped, Pose, TransformStamped
from tf.transformations import quaternion_from_euler
from copy import deepcopy
import numpy as np
import math
import tf2_ros
import tf
from tf2_geometry_msgs import PoseStamped as PoseStamp
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg


class MoveIt_Control:
    # 初始化程序
    def __init__(self, is_use_gripper=False):
        # Init ros config
        # rospy.init_node("test_control")
        moveit_commander.roscpp_initialize(sys.argv)

        self.arm = MoveGroupCommander('manipulator')
        self.arm.set_goal_joint_tolerance(0.001)
        self.arm.set_goal_position_tolerance(0.001)
        self.arm.set_goal_orientation_tolerance(0.01)

        self.end_effector_link = self.arm.get_end_effector_link()
        # 设置机械臂基座的参考系
        self.reference_frame = 'base'
        self.arm.set_pose_reference_frame(self.reference_frame)

        # 设置最大规划时间和是否允许重新规划
        self.arm.set_planning_time(5)
        self.arm.allow_replanning(True)
        self.arm.set_planner_id("RRTConnect")

        # 设置允许的最大速度和加速度（范围：0~1）
        self.arm.set_max_acceleration_scaling_factor(1)
        self.arm.set_max_velocity_scaling_factor(1)

        # 发布场景
        # self.set_scene()  # set table
        # self.arm.set_workspace([-2,-2,0,2,2,2])  #[minx miny minz maxx maxy maxz]
        self.move_j([0,0,0,0,0,0])
        self.move_p([0.5,0.5,0.5,np.pi,0,0])
        self.gripper_pub = rospy.Publisher(
            "Robotiq2FGripperRobotOutput", outputMsg.Robotiq2FGripper_robot_output, queue_size=1
        )
        
        

    # shutdown 函数
    def close(self):
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)

    # 限制规划的移动轨迹速度的函数
    def scale_trajectory_speed(self, plan,scale):
        n_joints = len(plan.joint_trajectory.joint_names)  # 关节数
        n_points = len(plan.joint_trajectory.points)   # 关节轨迹点数目
        print(n_joints)
        print(n_points)
        for i in range(n_points):
            plan.joint_trajectory.points[i].time_from_start *= 1/scale
            plan.joint_trajectory.points[i].velocities = list(np.array(plan.joint_trajectory.points[i].velocities)*scale)
            plan.joint_trajectory.points[i].velocities = list(np.array(plan.joint_trajectory.points[i].accelerations)*scale*scale)
        return plan

    # 安装夹爪的位置
    def install(self):
        self.move_j([0.224, -0.6, -2.2, -1.90, -1.57, 1.79],a=0.1,v=0.1)

    # 准备阶段，将机械臂调整到抓取前的姿势
    def prepare(self,a=1,v=1):
        # self.move_j([0.2240230143070221, -1.8436509571471156, -0.9659557342529297,   # height 60cm position
        #               -1.9027382336058558, 1.5708088874816895, 1.7948417663574219],a=0.2,v=0.2)

        self.move_j([0.24094422161579132, -1.7899095020689906, -1.3017092943191528, 
                     -1.6018387279906214, 1.5698283910751343, 1.8091448545455933],a=0.2,v=0.2)

    def prepare_tracking(self,a=1,v=1):
        self.move_j([0.22540307831373685, -1.809073798904528, -1.526457795012278,    # heiht 40cm position
                     -1.3738581140924744, 1.5604313861624846, 1.789932275395823],a=0.2,v=0.2)


    # 测试程序用
    def testRobot(self):
        try:
            print("Test for robot...")
            self.move_p([0.3, 0.2, 0.1, np.pi, 0, np.pi],a=1,v=1)
            self.move_p([0.3, -0.4, 0.5, np.pi, 0, np.pi],a=1,v=1)
            self.move_p([0.4, 0, 0.301, np.pi, 0, np.pi],a=1,v=1)

        except:
            print("Test fail! ")

  
    def cur_p(self):
        link_end = tf.transformations.quaternion_matrix([self.arm.get_current_pose().pose.orientation.x, 
                                                              self.arm.get_current_pose().pose.orientation.y, 
                                                              self.arm.get_current_pose().pose.orientation.z, 
                                                              self.arm.get_current_pose().pose.orientation.w])
        link_end[0,3] = self.arm.get_current_pose().pose.position.x
        link_end[1,3] = self.arm.get_current_pose().pose.position.y
        link_end[2,3] = self.arm.get_current_pose().pose.position.z
        z_rotate = np.mat([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
        base_end = np.dot(z_rotate,link_end)
        # print(base_end)
        (roll,pitch,yaw) = tf.transformations.euler_from_matrix(base_end)
        if roll < 0:
            roll = roll + 2 * np.pi
        if pitch < 0:
            pitch = pitch + 2 * np.pi
        if yaw < 0:
            yaw = yaw + 2 * np.pi
        cur_pose = [base_end[0,3],base_end[1,3],base_end[2,3],roll,pitch,yaw]
        return cur_pose
    
    # def cur_po(self):
    #     return self.arm.get_current_pose()

    def cur_j(self):
        return self.arm.get_current_joint_values()
    
    def log_pos(self):
        rospy.loginfo_throttle(1, self.cur_p())
        rospy.loginfo_throttle(1, [round(j, 3) for j in self.cur_j()])
        rospy.loginfo_throttle(1, [round(theta, 3) for theta in self.group.get_current_rpy()])

    # robotiq height and width
    def width2height(self,width):
        theta = math.asin((width+0.0156-0.025082)/0.12)
        height = 0.060 * math.cos(theta) + 0.103
        return height

    # 将相机坐标系转到base坐标系
    def tf_eye_base(self,rpypose):
        # 末端坐标系转到基座坐标系
        pose_rotation = tf.transformations.euler_matrix(rpypose.roll,rpypose.pitch,rpypose.yaw)
        pose_rotation[0,3] = rpypose.x + 0.032   # 0.032  # 0.020
        pose_rotation[1,3] = rpypose.y + 0.083  # 0.083   # 0.120
        pose_rotation[2,3] = rpypose.z + 0.079 - self.width2height(rpypose.width)
        end_cur = self.cur_p()
        # print(end_cur)
        # end_cur = [0.6, 0, 0.6, np.pi, 0, np.pi]
        # end_cur = [0.6, 0, 0.4, np.pi, 0, np.pi]
        transform_matrix = tf.transformations.euler_matrix(end_cur[3],end_cur[4],end_cur[5])
        transform_matrix[0,3] = end_cur[0]
        transform_matrix[1,3] = end_cur[1]
        transform_matrix[2,3] = end_cur[2]
        # 得到了目标位置
        target_pose = np.dot(transform_matrix,pose_rotation)
        target = {'x':0,'y':0,'z':0,'roll':0,'pitch':0,'yaw':0,'width':0}
        target["x"] = target_pose[0,3]
        target["y"] = target_pose[1,3]
        target["z"] = target_pose[2,3]
        target_pose = np.delete(target_pose, 3, axis=0)
        target_rotation = np.delete(target_pose, 3, axis=1)
        (target["roll"],target["pitch"],target["yaw"]) = tf.transformations.euler_from_matrix(target_rotation)
        target["width"] = rpypose.width
        return target



    # 在机械臂下方添加一个table，使得机械臂只能够在上半空间进行规划和运动
    # 避免碰撞到下方的桌子等其他物体
    def set_scene(self):
        ## set table
        self.scene = PlanningSceneInterface()
        self.scene_pub = rospy.Publisher('planning_scene', PlanningScene, queue_size=5)
        self.colors = dict()
        # rospy.sleep(1)
        ground_id = 'ground'
        self.scene.remove_world_object(ground_id)
        # rospy.sleep(1)
        ground_size = [2, 2, 0.01]
        ground_pose = PoseStamped() 
        ground_pose.header.frame_id = 'world'
        ground_pose.pose.position.x = 0.0
        ground_pose.pose.position.y = 0.0
        ground_pose.pose.position.z = 0.0
        ground_pose.pose.orientation.w = 1.0
        self.scene.add_box(ground_id, ground_pose, ground_size)
        self.setColor(ground_id, 0.5, 0.5, 0.5, 1.0)
        self.sendColors()

        base_table_id = 'ground'
        self.scene.remove_world_object(base_table_id)
        # rospy.sleep(1)
        base_table_size = [0.5, 0.5, 0.5]
        base_table_pose = PoseStamped()
        base_table_pose.header.frame_id = 'world'
        base_table_pose.pose.position.x = 0.0
        base_table_pose.pose.position.y = 0.0
        base_table_pose.pose.position.z = -base_table_size[2]/2
        base_table_pose.pose.orientation.w = 1.0
        self.scene.add_box(base_table_id, base_table_pose, base_table_size)
        self.setColor(base_table_id, 1.0, 0.5, 0.5, 1.0)
        self.sendColors()

        desk_id = 'desk'
        self.scene.remove_world_object(desk_id)
        # rospy.sleep(1)
        desk_size = [0.4, 0.8, 0.8]
        desk_pose = PoseStamped()
        desk_pose.header.frame_id = 'world'
        desk_pose.pose.position.x = 0.6+desk_size[0]/2
        desk_pose.pose.position.y = 0.0
        desk_pose.pose.position.z = -desk_size[2]/2
        desk_pose.pose.orientation.w = 1.0
        self.scene.add_box(desk_id, desk_pose, desk_size)
        self.setColor(desk_id, 0.5, 0.5, 1.0, 1.0)
        self.sendColors()


    # 关节规划，输入6个关节角度（单位：弧度）
    def move_j(self, joint_configuration=None,a=1,v=1):
        # 设置机械臂的目标位置，使用六轴的位置数据进行描述（单位：弧度）
        if joint_configuration==None:
            joint_configuration = [0, -1.5707, 0, -1.5707, 0, 0]
        self.arm.set_max_acceleration_scaling_factor(a)  # 设置最大加速度范围参数
        self.arm.set_max_velocity_scaling_factor(v)      # 设置最大速度范围参数
        self.arm.set_start_state_to_current_state()
        self.arm.set_joint_value_target(joint_configuration)
        rospy.loginfo("move_j:"+str(joint_configuration))
        self.arm.go()
        self.arm.stop()
        rospy.sleep(1)

    # 空间规划，输入xyzRPY
    def move_p(self, tool_configuration=None,a=1,v=1):
        if tool_configuration==None:
            tool_configuration = [0.3,0,0.3,0,-np.pi/2,0]
        self.arm.set_max_acceleration_scaling_factor(a)
        self.arm.set_max_velocity_scaling_factor(v)

        target_pose = PoseStamped()  # 位姿的消息格式
        target_pose.header.frame_id = self.reference_frame
        target_pose.header.stamp = rospy.Time.now()
        target_pose.pose.position.x = tool_configuration[0]
        target_pose.pose.position.y = tool_configuration[1]
        target_pose.pose.position.z = tool_configuration[2]
        q = quaternion_from_euler(tool_configuration[3],tool_configuration[4],tool_configuration[5]) # 欧拉角转换成四元数
        target_pose.pose.orientation.x = q[0]
        target_pose.pose.orientation.y = q[1]
        target_pose.pose.orientation.z = q[2]
        target_pose.pose.orientation.w = q[3]
        self.arm.set_pose_target(target_pose, self.end_effector_link)
        success = self.arm.go()
        # joints = self.arm.get_joint_value_target()
        # print(joints)
        rospy.loginfo("move_p:"+ " " + str(success) + " " + str(tool_configuration))
        # traj = self.arm.plan()
        # self.arm.execute(traj)
        # rospy.sleep(1)
    

    # 空间直线运动，输入(x,y,z,R,P,Y,x2,y2,z2,R2,...)
    # 默认仅执行一个点位，可以选择传入多个点位
    def move_l(self, tool_configuration,waypoints_number=1,a=0.02,v=0.02):
        if tool_configuration==None:
            tool_configuration = [0.3,0,0.3,0,-np.pi/2,0]
        # self.arm.set_max_acceleration_scaling_factor(a)
        # self.arm.set_max_velocity_scaling_factor(v)

        # 设置路点
        waypoints = []
        for i in range(waypoints_number):
            target_pose = PoseStamped()
            target_pose.header.frame_id = self.reference_frame
            target_pose.header.stamp = rospy.Time.now()
            target_pose.pose.position.x = tool_configuration[6*i+0]
            target_pose.pose.position.y = tool_configuration[6*i+1]
            target_pose.pose.position.z = tool_configuration[6*i+2]
            q = quaternion_from_euler(tool_configuration[6*i+3],tool_configuration[6*i+4],tool_configuration[6*i+5])
            target_pose.pose.orientation.x = q[0]
            target_pose.pose.orientation.y = q[1]
            target_pose.pose.orientation.z = q[2]
            target_pose.pose.orientation.w = q[3]
            waypoints.append(target_pose.pose)
        rospy.loginfo("move_l:" + str(tool_configuration))
        self.arm.set_start_state_to_current_state()
        fraction = 0.0  # 路径规划覆盖率
        maxtries = 100  # 最大尝试规划次数
        attempts = 0  # 已经尝试规划次数

        # 设置机器臂当前的状态作为运动初始状态
        self.arm.set_start_state_to_current_state()

        # 尝试规划一条笛卡尔空间下的路径，依次通过所有路点
        while fraction < 1.0 and attempts < maxtries:
            (plan, fraction) = self.arm.compute_cartesian_path(
                waypoints,  # waypoint poses，路点列表
                0.001,  # eef_step，终端步进值
                0.00,  # jump_threshold，跳跃阈值
                True)  # avoid_collisions，避障规划
            attempts += 1
            # print(fraction)
        if fraction == 1.0:
            rospy.loginfo("Path computed successfully. Moving the arm.")
            # print(plan.joint_trajectory.points[0].velocities[0]*0.1)
            # self.scale_trajectory_speed(plan,0.1)

            scale = 0.3

            plan = self.scale_trajectory_speed(plan,scale)

            self.arm.execute(plan)
            rospy.loginfo("Path execution complete.")
        else:
            rospy.loginfo(
                "Path planning failed with only " + str(fraction) +
                " success after " + str(maxtries) + " attempts.")
        rospy.sleep(1)

    def move_c(self,pose_via,tool_configuration,k_acc=1,k_vel=1,r=0,mode=0):
        pass

    # 控制夹爪开闭的函数
    def gripper(self,width):
        # 初始化ROS节点
        # rospy.init_node("Robotiq_close_open")
        # 创建发布者对象
        
        command = outputMsg.Robotiq2FGripper_robot_output()
        command.rACT = 1
        command.rGTO = 1
        command.rSP = 255
        command.rFR = 150

        max_width = 0.085
        command.rPR = int(255 * (1-width/max_width))

        for i in range(3):
            self.gripper_pub.publish(command)
            rospy.sleep(0.1)

    # 回到机械臂初始趴卧状态的函数
    def go_home(self,a=1,v=1):
        self.arm.set_max_acceleration_scaling_factor(a)
        self.arm.set_max_velocity_scaling_factor(v)
        # “up”为自定义姿态，你可以使用“home”或者其他姿态
        self.arm.set_named_target('home')
        self.arm.go()
        rospy.sleep(1)
    
    # 在set_scene()函数中使用
    def setColor(self, name, r, g, b, a=0.9):
        # 初始化moveit颜色对象
        color = ObjectColor()
        # 设置颜色值
        color.id = name
        color.color.r = r
        color.color.g = g
        color.color.b = b
        color.color.a = a
        # 更新颜色字典
        self.colors[name] = color

    # 在set_scene()函数中使用
    # 将颜色设置发送并应用到moveit场景当中
    def sendColors(self):
        # 初始化规划场景对象
        p = PlanningScene()
        # 需要设置规划场景是否有差异
        p.is_diff = True
        # 从颜色字典中取出颜色设置
        for color in self.colors.values():
            p.object_colors.append(color)
        # 发布场景物体颜色设置
        self.scene_pub.publish(p)
    

    def some_useful_function_you_may_use(self):
        # return the robot current pose
        current_pose = self.arm.get_current_pose()
        # rospy.loginfo('current_pose:',current_pose)
        # return the robot current joints
        current_joints = self.arm.get_current_joint_values()
        # rospy.loginfo('current_joints:',current_joints)

        #self.arm.set_planner_id("RRTConnect")
        self.arm.set_planner_id("TRRT")
        plannerId = self.arm.get_planner_id()
        rospy.loginfo(plannerId)

        planning_frame = self.arm.get_planning_frame()
        rospy.loginfo(planning_frame)

        # stop the robot
        self.arm.stop()
        

if __name__ =="__main__":
    rospy.init_node('moveit_planner')
    moveit_server = MoveIt_Control(is_use_gripper=False)

    
