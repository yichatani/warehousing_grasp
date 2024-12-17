# 打开相机节点
roslaunch azure_kinect_ros_driver driver.launch

# 仿真ur5e机械臂
roslaunch ur_gazebo ur5e_bringup.launch

roslaunch ur5e_moveit_config ur5e_moveit_planning_execution.launch limited:=true sim:=true

roslaunch ur5e_moveit_config moveit_rviz.launch config:=true

planner.py or demo_planner.py



# robotiq节点
dmesg | grep tty # 查看连接的串口
roscore
rosrun robotiq_2f_gripper_control Robotiq2FGripperRtuNode.py /dev/ttyUSB0
rosrun robotiq_2f_gripper_control Robotiq2FGripperSimpleController.py


# 控制实际ur5e机械臂
#1.启动驱动程序
roslaunch ur_robot_driver ur5e_bringup.launch robot_ip:=192.168.56.101
#在Ur5e面板上，点击【Run】，选择【加载文件】，选择上面自己保存的external_control.urp文件，底栏是有一个【播放按钮】的，按下就可以开始运行程序

#2.打开新终端，启动moveit
roslaunch ur5e_moveit_config ur5e_moveit_planning_execution.launch limited:=true

#3.打开新终端，启动rviz
roslaunch ur5e_moveit_config moveit_rviz.launch config:=true

#4.运行自编的规划程序
demo_planner.py