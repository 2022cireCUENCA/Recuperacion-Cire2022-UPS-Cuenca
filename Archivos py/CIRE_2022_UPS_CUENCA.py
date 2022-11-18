#!/usr/bin/env python3

__authors__ = "Andres Chacha, Luis Calle, Valeria Calle, Jordi Castel, Jose Sanchez, Kelvin Tigre, Juan Zumba" 
__group__ = "UPS Cuenca" 
__version__ = "1.0.0" 
__status__ = "Production"
__description__ = "Programa de recuperacion Takeshi: Movimiento del brazo y cabeza mediante moveit"

#--------------------------------- Librerias ------------------------------------------

import rospy
import numpy as np
import time
import tf2_ros 
import tf2_ros as tf2
import tf
import math
import smach
import ros_numpy
import matplotlib.pyplot as plt
import matplotlib
import sys
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import cv2
import os
import copy
import tmc_control_msgs.msg
import trajectory_msgs.msg

from gazebo_ros import gazebo_interface
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from math import atan2
from geometry_msgs.msg import PoseStamped,Pose
from utils_evasion import *
from sensor_msgs.msg   import LaserScan, PointCloud2
from geometry_msgs.msg import Twist, Point, Pose, Quaternion, TransformStamped, WrenchStamped
from utils_notebooks import *
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from std_msgs.msg import String



#------------------------------ Inicializando Nodo ------------------------------------

def init(node_name):
    global pub_cmd_vel, laser, base_vel_pub, listener1, broadcaster, rgbd,whole_body, head, arm, grip, tf_man,twist, rock_tf, x_goal, y_goal, theta_Goal
    rospy.init_node("Meta_etapa_5")
    base_vel_pub=rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=10) 
    pub_cmd_vel=rospy.Publisher("/hsrb/command_velocity", Twist, queue_size=10)
    twist=Twist()
    laser=Laser()
    rgbd = RGBD()
    listener1 = tf.TransformListener()
    tf_man = TF_MANAGER()
    grip = GRIPPER()
    broadcaster = tf.TransformBroadcaster()
    whole_body = moveit_commander.MoveGroupCommander('whole_body_light')
    head = moveit_commander.MoveGroupCommander('head')
    arm =  moveit_commander.MoveGroupCommander('arm')
    x_goal, y_goal = 0 , 0
#--------------------------------------Funcion movimiento cabeza---------------------------------------------
def gaze_point(x,y,z):
    head_pose = head.get_current_joint_values()
    head_pose[0]=0.0
    head_pose[1]=0.0
    head.set_joint_value_target(head_pose)
    head.go()
    trans , rot = listener1.lookupTransform('/map', '/head_rgbd_sensor_gazebo_frame', rospy.Time(0)) #
    e =tf.transformations.euler_from_quaternion(rot)
    x_rob,y_rob,z_rob,th_rob= trans[0], trans[1] ,trans[2] ,  e[2]
    D_x=x_rob-x
    D_y=y_rob-y
    D_z=z_rob-z
    D_th= np.arctan2(D_y,D_x)
    print('relative to robot',(D_x,D_y,np.rad2deg(D_th)))
    pan_correct= (- th_rob + D_th + np.pi) % (2*np.pi)
    if(pan_correct > np.pi):
        pan_correct=-2*np.pi+pan_correct
    if(pan_correct < -np.pi):
        pan_correct=2*np.pi+pan_correct
    if ((pan_correct) > .5 * np.pi):
        print ('Exorcist alert')
        pan_correct=.5*np.pi
    head_pose[0]=pan_correct
    tilt_correct=np.arctan2(D_z,np.linalg.norm((D_x,D_y)))
    head_pose [1]=-tilt_correct
    head.set_joint_value_target(head_pose)
    succ=head.go()
    return succ
    


#--------------------------------------Clase para transformadas de movimiento--------------------------------------------

class TF_MANAGER():
    def __init__(self):
        self._tfbuff = tf2.Buffer()
        self._lis = tf2.TransformListener(self._tfbuff)
        self._tf_static_broad = tf2.StaticTransformBroadcaster()
        self._broad = tf2.TransformBroadcaster()

    def _fillMsg(self, pos = [0,0,0], rot = [0,0,0,1] ,point_name ='', ref="odom"):
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

    def pub_tf(self, pos = [0,0,0], rot = [0,0,0,1] ,point_name ='', ref="odom"):
        dinamic_ts = self._fillMsg(pos, rot, point_name, ref)
        self._broad.sendTransform(dinamic_ts)

    def pub_static_tf(self, pos = [0,0,0], rot = [0,0,0,1] ,point_name ='', ref="odom"):
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
        
#----------------------------------Clase para movimiento del gripper--------------------------------------------
        
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
#--------------------------------- Puntos Objetos -------------------------------------
#aumentar estados a la variable actual_points
def points():
    punto_act=get_coords()
    xt=punto_act.transform.translation.x
    yt=punto_act.transform.translation.y
    (roll, pitch, theta) = euler_from_quaternion ([punto_act.transform.rotation.x, punto_act.transform.rotation.y, punto_act.transform.rotation.z, punto_act.transform.rotation.w])	
    x_goal,y_goal=1.3 * np.cos(theta),1.3 * np.sin(theta)
    Theta_goal=theta
    return x_goal, y_goal,Theta_goal
   
#------------------------- Funcion Acondicionamiento Laser ---------------------------
        
def get_lectura_cuant():
    try:
        global left_scan, right_scan, front_left_point, front_right_point
        lectura=np.asarray(laser.get_data().ranges)
        lectura=np.where(lectura>20,20,lectura) #remove infinito
        right_scan=lectura[:180] #180
        left_scan=lectura[540:] #540
        front_scan=lectura[180:540] #180:540
       # front_point=lectura[260:460]
        #front_right_point=lectura[100:260]
        #front_left_point=lectura[460:620]
        sd,si,sf=0,0,0
        if np.mean(left_scan)< 1.8: si=1
        if np.mean(right_scan)< 1.5: sd=1
        if np.mean(front_scan)< 1.8: sf=1
    except:
        sd,si,sf=0,0,0    
    return si,sd,sf
#---------------------------- Funcion Coordenadas ------------------------------------

def get_coords():
    for i in range(10): 
        try:
            trans=tfBuffer.lookup_transform('map', 'base_link', rospy.Time())
            return trans     
        except:
            trans=0    

#------------------------- Funciones Movimiento Robot --------------------------------

def move_base_vel(vx, vy, vw):
    twist.linear.x=vx
    twist.linear.y=vy
    twist.angular.z=vw 
    base_vel_pub.publish(twist)
def move_base(x,y,yaw,timeout):
    start_time=rospy.Time.now().to_sec()
    while rospy.Time.now().to_sec()-start_time<timeout:  
        move_base_vel(x, y, yaw) 
def stop_base():
    start_time=rospy.Time.now().to_sec()
    while rospy.Time.now().to_sec()-start_time<0.5:  
        move_base_vel(0, 0, 0)
 
 
#-------------------------Movimiento de brazo por articulaciones--------------------------------       
def goal_points(art0, art1, art2, art3, art4, art5 ):
    print("Moviendo")
    joint_goal = arm.get_current_joint_values()
    joint_goal[0] = art0
    joint_goal[1] = art1
    joint_goal[2] = art2
    joint_goal[3] = art3
    joint_goal[4] = art4
    joint_goal[5] = art5
    arm.set_joint_value_target(joint_goal)
    arm.go()
    print("Puntos Finales")

#---------------------------------- Estados --------------------------------------    

class S0(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1','outcome2','outcome3','outcome4','outcome5','outcome6','outcome7','outcome8','outcome9'])
        self.counter=0
    def execute(self,userdata):
        global theta_Goal,angle_to_goal, angle_to_goal1, xt, yt, inc_x, inc_y, theta, x_goal, y_goal
        
        si,sd,sf=get_lectura_cuant()
        msg_cmd_vel=Twist()
        punto_act=get_coords()
        xt=punto_act.transform.translation.x
        yt=punto_act.transform.translation.y
        (roll, pitch, theta) = euler_from_quaternion ([punto_act.transform.rotation.x, punto_act.transform.rotation.y, punto_act.transform.rotation.z, punto_act.transform.rotation.w])	
        inc_x=x_goal-xt
        inc_y=y_goal-yt
        angle_to_goal=atan2(inc_y, inc_x)
        angle_to_goal1=angle_to_goal-theta
        
        '''if (np.mean(front_left_point)<1.5):
            print('base apegada izquierda')
            move_base(0.2,0.0,-0.12*np.pi,1)
            return 'outcome1'
        if (np.mean(front_right_point)<1.5):
            print('base apegada derecha')
            move_base(0.2,0.0,0.12*np.pi,1)
            return 'outcome1'''
        
        if (si==0 and sd==0 and sf==0): return 'outcome9'     
        if (si==0 and sf==0 and sd==1): return 'outcome2'
        if (si==0 and sf==1 and sd==0): return 'outcome3'
        if (si==0 and sf==1 and sd==1): return 'outcome4'
        if (si==1 and sf==0 and sd==0): return 'outcome5'
        if (si==1 and sf==0 and sd==1): return 'outcome6'
        if (si==1 and sf==1 and sd==0): return 'outcome7'
        if (si==1 and sf==1 and sd==1): return 'outcome8'
        return 'outcome1' 
        pub_cmd_vel.publish(msg_cmd_vel)
        r.sleep() 

class S1(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1','outcome2'])
        self.counter=0
    def execute(self,userdata):
        rospy.loginfo('Estado Derecha')
        if abs(inc_x)>0.05 and abs(inc_y)>0.05:
            move_base(0.3,0.0,0.12*np.pi,0.08)
        return 'outcome1'

class S2(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1','outcome2'])
        self.counter=0
    def execute(self,userdata):
        rospy.loginfo('Estado Frente')
        if abs(inc_x)>0.05 and abs(inc_y)>0.05:
            if ((angle_to_goal1>=0 and angle_to_goal1<np.pi) or (angle_to_goal1<-np.pi)):
                if abs(angle_to_goal1)>0.1: move_base(0.0,0.0,0.12*np.pi,0.1)                 
            elif ((angle_to_goal1>=np.pi and angle_to_goal1<2*np.pi) or (angle_to_goal1<0 and angle_to_goal1>-(np.pi))):
                if abs(angle_to_goal1)>0.1: move_base(0.0,0.0,-0.12*np.pi,0.1)  
        return 'outcome1'

class S3(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1','outcome2'])
        self.counter=0
    def execute(self,userdata):
        rospy.loginfo('Estado Frente - Derecha')
        if abs(inc_x)>0.05 and abs(inc_y)>0.05:
            move_base(0.0,0.0,0.12*np.pi,0.1)
        return 'outcome1'

class S4(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1','outcome2'])
        self.counter=0
    def execute(self,userdata):
        rospy.loginfo('Estado Izquierda')
        if abs(inc_x)>0.05 and abs(inc_y)>0.05:
            move_base(0.0,0.0,-0.12*np.pi,0.8)
            move_base(0.2,0.0,0.0,0.8)
        return 'outcome1' 

class S5(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1','outcome2'])
        self.counter=0
    def execute(self,userdata):
        rospy.loginfo('Estado Izquierda - Derecha')
        move_base(0.3,0,0,0.1)
        return 'outcome1' 

class S6(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1','outcome2'])
        self.counter=0
    def execute(self,userdata):
        rospy.loginfo('Estado Frente - Izquierda')
        if abs(inc_x)>0.05 and abs(inc_y)>0.05:
            move_base(0.0,0.0,-0.12*np.pi,0.1)
        return 'outcome1' 

class S7(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1','outcome2'])
        self.counter=0
    def execute(self,userdata):
        rospy.loginfo('Estado Reversa')
        move_base(-0.3,0,0,0.1)
        return 'outcome1' 

class S8(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1','outcome2'])
        self.counter=0
    def execute(self,userdata):
        rospy.loginfo('Estado Comparacion')
        global angle_to_goal1, theta_Goal, x_goal,y_goal, aux, num_rock,rock_tf, indice_r
        if abs(inc_x)<0.05 and abs(inc_y)<0.05: 
            
            angle_to_goal_p=theta_Goal-theta 
            if ((angle_to_goal_p >= 0 and angle_to_goal_p <np.pi) or (angle_to_goal_p <-np.pi)):
                if abs(angle_to_goal1)>0.1: move_base(0.0,0.0,0.12*np.pi,0.03)  #move_base(0.0,0.0,theta_Goal,0.1)     
            elif ((angle_to_goal_p >=np.pi and angle_to_goal_p <2*np.pi) or (angle_to_goal_p < 0 and angle_to_goal_p > -(np.pi) )):
                if abs(angle_to_goal1)>0.1: move_base(0.0,0.0,-0.12*np.pi,0.03)  #move_base(0.0,0.0,-theta_Goal,0.1) # 
            
            if abs(angle_to_goal_p) <0.1:
                punto_fin=get_coords()
                (roll_final, pitch_final, theta_final) = euler_from_quaternion ([punto_fin.transform.rotation.x, punto_fin.transform.rotation.y, punto_fin.transform.rotation.z, punto_fin.transform.rotation.w])
                print('Tiempo = '+ str(punto_fin.header.stamp.to_sec()))
                print("x final: " + str(punto_fin.transform.translation.x))
                print("y final: " + str(punto_fin.transform.translation.y))
                print("Theta Final: " + str(theta_final))
                whole_body.stop()
                rospy.sleep(5)
                goal_points(0.5,0.0,0.0,0.0,0.0,0.0)
                rospy.sleep(1)
                goal_points(0.5,-0.7,2.0,0.78,0.0,0.0)
                rospy.sleep(1)
                head.go(np.array((-0.15*np.pi,0)))
                rospy.sleep(1)
                head.go(np.array(( 0.15*np.pi,0)))
                
                return 'outcome2'
                
            
        else:
            if ((angle_to_goal1>=0 and angle_to_goal1<np.pi) or (angle_to_goal1<-np.pi)):
                if abs(angle_to_goal1)>0.1: move_base(0.0,0.0,0.12*np.pi,0.1)                 
                if abs(angle_to_goal1)<0.1: move_base(0.3,0,0,0.1)
            elif ((angle_to_goal1>=np.pi and angle_to_goal1<2*np.pi) or (angle_to_goal1<0 and angle_to_goal1>-(np.pi))):
                if abs(angle_to_goal1)>0.1: move_base(0.0,0.0,-0.12*np.pi,0.1)  
                if abs(angle_to_goal1)<0.1: move_base(0.3,0,0,0.1)
        return 'outcome1'

def main():
    global x_goal, y_goal, theta_Goal
    EQUIPO="UPS Cuenca"
    print("Meta Competencia Etapa 5 - " + EQUIPO)
    print('Inicializando')
    r=rospy.Rate(4)	
    rospy.sleep(1)
    punto_ini=get_coords()
    print ('Tiempo Inicial = '+ str(punto_ini.header.stamp.to_sec()))
    #head.go(np.array((0,-.15*np.pi)))
    #arm.set_named_target('go')
    #arm.go()
    x_goal,y_goal,theta_Goal=points()
    
    
    
if __name__ == '__main__':
    #init("Meta_etapa_5")
    init("takeshi_smach")
    sm=smach.StateMachine(outcomes=['END'])     #State machine, final state "END"
    sm.userdata.sm_counter=0
    sm.userdata.clear=False   
    with sm:
        smach.StateMachine.add("s_0",   S0(),  transitions = {'outcome1':'s_0', 'outcome2':'s_1','outcome3':'s_2','outcome4':'s_3','outcome5':'s_4', 'outcome6':'s_5','outcome7':'s_6','outcome8':'s_7','outcome9':'s_8',})
        smach.StateMachine.add("s_1",   S1(),  transitions = {'outcome1':'s_8','outcome2':'END'})
        smach.StateMachine.add("s_2",   S2(),  transitions = {'outcome1':'s_8','outcome2':'END'})
        smach.StateMachine.add("s_3",   S3(),  transitions = {'outcome1':'s_8','outcome2':'END'})
        smach.StateMachine.add("s_4",   S4(),  transitions = {'outcome1':'s_8','outcome2':'END'})
        smach.StateMachine.add("s_5",   S5(),  transitions = {'outcome1':'s_8','outcome2':'END'})
        smach.StateMachine.add("s_6",   S6(),  transitions = {'outcome1':'s_8','outcome2':'END'})
        smach.StateMachine.add("s_7",   S7(),  transitions = {'outcome1':'s_2','outcome2':'END'})
        smach.StateMachine.add("s_8",   S8(),  transitions = {'outcome1':'s_0','outcome2':'END'})
    try:
        tfBuffer=tf2_ros.Buffer()
        listener=tf2_ros.TransformListener(tfBuffer)
        main()
    except rospy.ROSInterruptException:
        pass
        
outcome=sm.execute()
    
