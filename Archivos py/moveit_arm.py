#!/usr/bin/env python3

#--------------------------------- Librerias ------------------------------------------
import sys
import copy
import rospy
import moveit_commander ##So important to run moveit
import moveit_msgs.msg ##So important to run moveit
import geometry_msgs.msg
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

import tf as tf
import tf2_ros as tf2
import rospy
import numpy as np
import ros_numpy
from std_msgs.msg import String
from geometry_msgs.msg import Twist, WrenchStamped, TransformStamped
import tmc_control_msgs.msg
import trajectory_msgs.msg
import math
from nav_msgs.msg import Odometry
#------------------------------ Inicializando Nodo ------------------------------------


moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node("Tutorial_moveit", anonymous=True)
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
EQUIPO="UPS Cuenca"
r=rospy.Rate(4)
twist=Twist()
display_trajectory_publisher = rospy.Publisher("/move_group/display_planned_path",    moveit_msgs.msg.DisplayTrajectory,queue_size=20,)
listener = None

head = moveit_commander.MoveGroupCommander('head')
whole_body = moveit_commander.MoveGroupCommander('whole_body_light')
arm =  moveit_commander.MoveGroupCommander('arm')
gripper =  moveit_commander.MoveGroupCommander('gripper')

global points
points=0;

def init(node_name):
    global laser, base_vel_pub
    base_vel_pub = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=10) 
    

#---------------------------------- Control -------------------------------------- 
 
def calculate_control(robot_x, robot_y, robot_a, goal_x, goal_y):
    cmd_vel = Twist()    
    v_max = 0.3
    w_max = 0.5
    alpha = 1.0#0.2
    beta  = 0.1#0.5
    [error_x, error_y] = [goal_x - robot_x, goal_y - robot_y]
    error_a = (math.atan2(error_y, error_x) - robot_a)%(2*math.pi)
    error_d = math.sqrt(error_x**2 + error_y**2)
    if error_a  >  math.pi:
        error_a -= 2*math.pi
    cmd_vel.linear.x  = min(v_max, error_d)*math.exp(-error_a*error_a/alpha)
    cmd_vel.angular.z = w_max*(2/(1 + math.exp(-error_a/beta)) - 1)
    return cmd_vel

#---------------------------------- Robot Pose --------------------------------------

def get_robot_pose(listener):
    try:
        ([x, y, z], rot) = listener.lookupTransform('map', 'base_link', rospy.Time(0))
        a = 2*math.atan2(rot[2], rot[3])
        a = a - 2*math.pi if a > math.pi else a
        return [x, y, a]
    except:
        pass
    return [0,0,0]


class TF_MANAGER():
    def __init__(self):
        self._tfbuff = tf2.Buffer()
        self._lis = tf2.TransformListener(self._tfbuff)
        self._tf_static_broad = tf2.StaticTransformBroadcaster()
        self._broad = tf2.TransformBroadcaster()

    def _fillMsg(self, pos = [0,0,0], rot = [0,0,0,1] ,point_name ='', ref="map"):
        TS = TransformStamped()
        TS.header.stamp = rospy.Time.now()
        TS.header.frame_id = ref
        TS.child_frame_id = point_name
        TS.transform.translation.x = pos[0]
        TS.transform.translation.y = pos[1]
        TS.transform.translation.z = pos[2]
        TS.transform.rotation.x = rot[0]
        TS.transform.rotation.y = rot[1]
        TS.transform.rotation.z = rot[2]
        TS.transform.rotation.w = rot[3]
        return TS

    def pub_tf(self, pos = [0,0,0], rot = [0,0,0,1] ,point_name ='', ref="map"):
        dinamic_ts = self._fillMsg(pos, rot, point_name, ref)
        self._broad.sendTransform(dinamic_ts)

    def pub_static_tf(self, pos = [0,0,0], rot = [0,0,0,1] ,point_name ='', ref="map"):
        static_ts = self._fillMsg(pos, rot, point_name, ref)
        self._tf_static_broad.sendTransform(static_ts)

    def change_ref_frame_tf(self, point_name = '', new_frame = 'map'):
        try:
            traf = self._tfbuff.lookup_transform(new_frame, point_name, rospy.Time(0))
            translation, rotational = self.tf2_obj_2_arr(traf)
            self.pub_static_tf(pos = translation, rot = rotational, point_name = point_name, ref = new_frame)
            return True
        except:
            return False

    def getTF(self, target_frame='', ref_frame='map'):
        try:
            tf = self._tfbuff.lookup_transform(ref_frame, target_frame, rospy.Time(0))
            return self.tf2_obj_2_arr(tf)
        except:
            return [False,False]

    def tf2_obj_2_arr(self, transf):
        pos = []
        pos.append(transf.transform.translation.x)
        pos.append(transf.transform.translation.y)
        pos.append(transf.transform.translation.z)
    
        rot = []
        rot.append(transf.transform.rotation.x)
        rot.append(transf.transform.rotation.y)
        rot.append(transf.transform.rotation.z)
        rot.append(transf.transform.rotation.w)

        return [pos, rot]
class GRIPPER():
    def __init__(self):
        self._grip_cmd_pub = rospy.Publisher('/hsrb/gripper_controller/command',
                               trajectory_msgs.msg.JointTrajectory, queue_size=100)
        self._grip_cmd_force = rospy.Publisher('/hsrb/gripper_controller/grasp/goal',
        			tmc_control_msgs.msg.GripperApplyEffortActionGoal, queue_size=100)
        			
        self._joint_name = "hand_motor_joint"
        self._position = 0.5
        self._velocity = 0.5
        self._effort = 0.0
        self._duration = 1

    def _manipulate_gripper(self):
        traj = trajectory_msgs.msg.JointTrajectory()
        traj.joint_names = [self._joint_name]
        p = trajectory_msgs.msg.JointTrajectoryPoint()
        p.positions = [self._position]
        p.velocities = [self._velocity]
        p.accelerations = []
        p.effort = [self._effort]
        p.time_from_start = rospy.Duration(self._duration)
        traj.points = [p]
        self._grip_cmd_pub.publish(traj)
        
    def _apply_force(self):
        app_force = tmc_control_msgs.msg.GripperApplyEffortActionGoal()
        app_force.goal.effort = -0.5
        self._grip_cmd_force.publish(app_force)
        
    def change_velocity(self, newVel):
        self._velocity = newVel
    
    def open(self):
        self._position = 1.23
        self._effort = 0
        self._manipulate_gripper()

    def steady(self):
        self._position = -0.82
        self._effort = -0.3
        self._manipulate_gripper()
        
    def close(self):
        self._position = -0.82
        self._effort = -0.3
        self._manipulate_gripper()
        self._apply_force()
        rospy.sleep(0.8)

def plane_reference():
    arm_planning_frame = arm.get_planning_frame()
    head_planning_frame = head.get_planning_frame()
    wb_planning_frame = whole_body.get_planning_frame()

    print(f">>>Planning frame: {arm_planning_frame}, {head_planning_frame}, {wb_planning_frame}")

    # We can also print the name of the end-effector link for this group:
    eef_link = arm.get_end_effector_link()
    print(f">>>End effector link: {eef_link}")

def hand_palm_link(arm_points,arm_x,arm_y,arm_z):
    
    rot, trans = False, False
    tf_man = TF_MANAGER()
    
    while(not rot):
        _, rot = tf_man.getTF(target_frame='hand_palm_link', ref_frame='odom')
    roll,pitch,yaw = np.pi,0, 0.9
    rot1 = tf.transformations.quaternion_from_euler(roll,pitch,yaw)
    tf_man.pub_static_tf(pos=[arm_x,arm_y,arm_z], rot = rot1, point_name='goal', ref='odom')
    rot = False
    print("trans_1: " + str(trans))
    print("rot_1: " + str(rot))
    while( not rot or not trans):
        trans, rot = tf_man.getTF(target_frame='goal', ref_frame='odom')
        print("trans: " + str(trans))
        print("rot: " + str(rot))
        rospy.sleep(0.2)
    #Inverse Kinematics
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.orientation.w = rot1[3]
    pose_goal.orientation.x = rot1[0]
    pose_goal.orientation.y = rot1[1]
    pose_goal.orientation.z = rot1[2]

    pose_goal.position.x = arm_x
    pose_goal.position.y = arm_y
    pose_goal.position.z = arm_z
    whole_body.set_start_state_to_current_state()
    whole_body.set_pose_target(pose_goal)   
    whole_body.go(wait=True)
    whole_body.stop()
    whole_body.clear_pose_targets()

def arm_go_mode():
    arm.set_named_target('go') 
    arm.go()

def arm_neutral_mode():
    arm.set_named_target('neutral') 
    arm.go()

def main():
    global pub_cmd_vel, detect,ini,points
    tf_man = TF_MANAGER()
    print("Meta Competencia - " + EQUIPO)
    pub_cmd_vel=rospy.Publisher("/hsrb/command_velocity", Twist, queue_size=10)
    listener = tf.TransformListener()
    print('Inicializando')
    grip = GRIPPER()
    rospy.sleep(1)
    msg_cmd_vel=Twist()
    #goal_points()
    robot_x, robot_y, robot_a = get_robot_pose(listener)
    points = 0
    arm = 0
    arm_go_mode()
    while (1):
        if (points==0 ):
            
            robot_x, robot_y, robot_a = get_robot_pose(listener)
            msg_cmd_vel = calculate_control(robot_x, robot_y, robot_a,1,-1)
            pub_cmd_vel.publish(msg_cmd_vel)
            print("robot_x "+str(robot_x))
            print("robot_y "+str(robot_y))
            if (abs(robot_x - 0.40204205052699715 < 1) and abs(robot_y - 0.9365304683317035 ) < 1):
                points = 1
                arm=1
        print("Llego")    
        while(arm == 1):
            print("While")
            grip.open()
            hand_palm_link(points,0.40204205,0.93653047,0.03785152)
            rospy.sleep(1)
            grip.close()
            arm_neutral_mode()
            arm.go()
            if (points==1 ):
                robot_x, robot_y, robot_a = get_robot_pose(listener)
                msg_cmd_vel = calculate_control(robot_x, robot_y, robot_a,2.14 - 0.5 - 0.75 , 0.73)
                pub_cmd_vel.publish(msg_cmd_vel)
                print("robot_x "+str(robot_x))
                print("robot_y "+str(robot_y))
                if (abs(robot_x - 2.14 - 0.5 - 0.75 < 1) and abs(robot_y - 0.73 ) < 1):
                    points = 1
                    arm=1
            hand_palm_link(2.14,0.73,0.4)
            print("arm" + str(arm))
    
if __name__ == '__main__':
    init("takeshi_smach")
    
    try:
        main()
    except rospy.ROSInterruptException:
        pass
        

    
