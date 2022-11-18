#!/usr/bin/env python3

__authors__ = "Andres Chacha, Luis Calle, Valeria Calle, Jordi Castel, Jose Sanchez, Kelvin Tigre, Juan Zumba" 
__group__ = "UPS Cuenca" 
__version__ = "1.0.0" 
__status__ = "Production"
__description__ = "Programa etapa 5: Reconocimiento de objetos y manipulacion del brazo antropomorfico para ubicacion de rocas"

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

moveit_commander.roscpp_initialize(sys.argv)
display_trajectory_publisher = rospy.Publisher("/move_group/display_planned_path",    moveit_msgs.msg.DisplayTrajectory,queue_size=20,)

def init(node_name):
    global Theta_goal, pub_cmd_vel, laser, base_vel_pub, next_point,listener1, broadcaster, rgbd,whole_body, head, arm, grip, tf_man,twist, rock_tf, x_goal, y_goal, theta_Goal, aux, num_rock
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
    next_point=1
    rock_tf = []
    x_goal, y_goal = 0 , 0
    aux = 0
    num_rock = 0
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
    
#--------------------------------------Transformadas de Objetos--------------------------------------------
def Obtener_datos_camara():
    aux = 0
    while (aux==0):
        image=rgbd.get_image() 
        points=rgbd.get_points()
        if np.any(image!= None):
            plt.imshow(image)
            #plt.show()
            aux=1
    im_bgr=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray_image=cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
    ret,thresh=cv2.threshold(gray_image,70,75,0)
    plt.imshow(thresh)
    #plt.show() 
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    centcamera = []
    centmap = []
    for c in contours:
        M=cv2.moments(c)
        #print("M00")
        #print(M["m00"])
        #print("M10")
        #print(M["m10"])
        if int(M["m00"])>200 and int(M["m00"])<1000 and int(M["m10"])>10000 and int(M["m10"])<900000:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            boundRect = cv2.boundingRect(c)
            image2=cv2.rectangle(thresh,(boundRect[0], boundRect[1]),(boundRect[0]+boundRect[2], boundRect[1]+boundRect[3]),(255,255,255),2)
            cv2.circle(image2, (cX, cY), 5, (255,255,255),-1)
            cv2.putText(image2, "centroid_"+str(cX)+','+str(cY) , (cX - 50, cY - 25) , cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255,255,255),2)
            plt.imshow(image2)
            #plt.show()
            xyz=[]  
            for jy in range (boundRect[0], boundRect[0]+boundRect[2]):
                for ix in range(boundRect[1], boundRect[1]+boundRect[3]):
                    aux=(np.asarray((points['x'][ix,jy],points['y'][ix,jy],points['z'][ix,jy])))
                    if np.isnan(aux[0]) or np.isnan(aux[1]) or np.isnan(aux[2]):
                        'reject point'
                    else:
                        xyz.append(aux)
            xyz=np.asarray(xyz)
            cent=xyz.mean(axis=0)
            centcamera = np.concatenate((centcamera, cent), axis=0)
            x,y,z=cent    
            if np.isnan(x) or np.isnan(y) or np.isnan(z):
                print('nan')
            else:
                broadcaster.sendTransform((x,y,z),(0,0,0,1), rospy.Time.now(), 'Object',"head_rgbd_sensor_link")
                time.sleep(5)
                tf_mapa=listener1.lookupTransform('map','Object',rospy.Time(0))
                #broadcaster.sendTransform((tf_mapa[0][0],tf_mapa[0][1],tf_mapa[0][2]),(0,0,0,1), rospy.Time.now(), 'Object_fix','map')
                auxtf = tf_mapa[0] 
                centmap = np.concatenate((centmap, auxtf), axis=0)
                #print(centmap)
    return centmap

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

def plane_reference():
    arm_planning_frame = arm.get_planning_frame()
    head_planning_frame = head.get_planning_frame()
    wb_planning_frame = whole_body.get_planning_frame()

    print(f">>>Planning frame: {arm_planning_frame}, {head_planning_frame}, {wb_planning_frame}")

    # We can also print the name of the end-effector link for this group:
    eef_link = arm.get_end_effector_link()
    print(f">>>End effector link: {eef_link}")        
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

#----------------------------------Funcion Movimeinto Brazo--------------------------------------------

def hand_palm_link(arm_points,arm_x,arm_y,arm_z):
    
    rot, trans = False, False
    tf_man = TF_MANAGER()
    
    while(not rot):
        _, rot = tf_man.getTF(target_frame='hand_palm_link', ref_frame='odom')
    roll,pitch,yaw = np.pi,0, 0.9
    rot1 = tf.transformations.quaternion_from_euler(roll,pitch,yaw)
    tf_man.pub_static_tf(pos=[arm_x,arm_y,arm_z], rot = rot1, point_name='goal', ref='odom')
    rot = False
    #print("trans_1: " + str(trans))
    #print("rot_1: " + str(rot))
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
    
#--------------------------------- Puntos Objetos -------------------------------------
#aumentar estados a la variable actual_points
def points(actual_point):
# primer centroide
    if actual_point==1:
        x_goal,y_goal=1,0
        #x_goal,y_goal=8,0 #0,1.21-1.1
        Theta_goal=np.pi*2/3 
        return x_goal, y_goal,Theta_goal
# segundo centroide        
    elif actual_point==2:
        x_goal, y_goal = -1, 4  #-3.0, 4.0-1.1
        Theta_goal=np.pi
        return x_goal, y_goal,Theta_goal
#Tercer centroide
    elif actual_point==3:
        x_goal,y_goal= 2,5  #3.9-1.1, 5.6
        Theta_goal=np.pi/9 
        return x_goal,y_goal,Theta_goal
#Punto home
    elif actual_point==4:
        x_goal,y_goal= 0,0  
        Theta_goal=0 
        return x_goal,y_goal,Theta_goal
    
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
#---------------------------------- Estados --------------------------------------    

class S0(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1','outcome2','outcome3','outcome4','outcome5','outcome6','outcome7','outcome8','outcome9'])
        self.counter=0
    def execute(self,userdata):
        global theta_Goal,angle_to_goal, angle_to_goal1, xt, yt, inc_x, inc_y, theta, x_goal, y_goal
        if (x_goal != 2.14 - 0.5 - 0.75 and y_goal != 0.73):
            x_goal,y_goal,theta_Goal=points(next_point)
            
        #print('Punto ' + str(next_point))
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
        #rospy.loginfo('Estado Derecha')
        if abs(inc_x)>0.05 and abs(inc_y)>0.05:
            move_base(0.3,0.0,0.12*np.pi,0.08)
        return 'outcome1'

class S2(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1','outcome2'])
        self.counter=0
    def execute(self,userdata):
        #rospy.loginfo('Estado Frente')
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
        #rospy.loginfo('Estado Frente - Derecha')
        if abs(inc_x)>0.05 and abs(inc_y)>0.05:
            move_base(0.0,0.0,0.12*np.pi,0.1)
        return 'outcome1'

class S4(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1','outcome2'])
        self.counter=0
    def execute(self,userdata):
        #rospy.loginfo('Estado Izquierda')
        if abs(inc_x)>0.05 and abs(inc_y)>0.05:
            move_base(0.0,0.0,-0.12*np.pi,0.8)
            move_base(0.2,0.0,0.0,0.8)
        return 'outcome1' 

class S5(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1','outcome2'])
        self.counter=0
    def execute(self,userdata):
        #rospy.loginfo('Estado Izquierda - Derecha')
        move_base(0.3,0,0,0.1)
        return 'outcome1' 

class S6(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1','outcome2'])
        self.counter=0
    def execute(self,userdata):
        #rospy.loginfo('Estado Frente - Izquierda')
        if abs(inc_x)>0.05 and abs(inc_y)>0.05:
            move_base(0.0,0.0,-0.12*np.pi,0.1)
        return 'outcome1' 

class S7(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1','outcome2'])
        self.counter=0
    def execute(self,userdata):
        #rospy.loginfo('Estado Reversa')
        move_base(-0.3,0,0,0.1)
        return 'outcome1' 

class S8(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['outcome1','outcome2'])
        self.counter=0
    def execute(self,userdata):
        #rospy.loginfo('Estado Comparacion')
        if abs(inc_x)<0.05 and abs(inc_y)<0.05: 
            global angle_to_goal1, next_point, theta_Goal, x_goal,y_goal, aux, num_rock,rock_tf, indice_r, arm
            angle_to_goal_p=theta_Goal-theta 
            if ((angle_to_goal_p >= 0 and angle_to_goal_p <np.pi) or (angle_to_goal_p <-np.pi)):
                if abs(angle_to_goal1)>0.1: move_base(0.0,0.0,0.12*np.pi,0.1)  #move_base(0.0,0.0,theta_Goal,0.1)     
            elif ((angle_to_goal_p >=np.pi and angle_to_goal_p <2*np.pi) or (angle_to_goal_p < 0 and angle_to_goal_p > -(np.pi) )):
                if abs(angle_to_goal1)>0.1: move_base(0.0,0.0,-0.12*np.pi,0.1)  #move_base(0.0,0.0,-theta_Goal,0.1) # 
            
            if abs(angle_to_goal_p) <0.1:
                punto_fin=get_coords()
                (roll_final, pitch_final, theta_final) = euler_from_quaternion ([punto_fin.transform.rotation.x, punto_fin.transform.rotation.y, punto_fin.transform.rotation.z, punto_fin.transform.rotation.w])
                #print('Tiempo = '+ str(punto_fin.header.stamp.to_sec()))
                print("x final: " + str(punto_fin.transform.translation.x))
                print("y final: " + str(punto_fin.transform.translation.y))
                #print("Theta Final: " + str(theta_final))
                
                whole_body.stop()
                
                rospy.sleep(5)
                '''if (x_goal == 2.14 - 0.5 - 0.75 and y_goal == 0.73):
                    hand_palm_link(2.14,0.73,0.4)
                    grip.open()
                    rospy.sleep(3)
                    hand_palm_link(2.14,0.73,0.7)
                    grip.close()
                    arm.set_named_target('go')
                    arm.go()
                    x_goal,y_goal,theta_Goal=points(next_point)
                
                else:
                    #print("Buscando")
                    whole_body.stop()
                    #time.sleep(1)'''
                if (next_point==1 or next_point==2 or next_point==3) :
                    time.sleep(5)
                    rock_tf = Obtener_datos_camara()
                    indice_r = 0
                    #num_rock = len(rock_tf)/3
                    #print("num_rock"+ str(num_rock))
                
                '''if num_rock != aux:
                    grip.open()
                    points = 0
                    rospy.sleep(2)
                    robot_x=2*rock_tf[indice_r]
                    robot_y=2*rock_tf[indice_r+1]
                    while(abs(robot_x - rock_tf[indice_r]) > 0.5 or abs(robot_y - rock_tf[indice_r+1]) > 0.5):
                        robot_x, robot_y, robot_a = get_robot_pose(listener1)
                        msg_cmd_vel = calculate_control(robot_x, robot_y, robot_a,rock_tf[indice_r],rock_tf[indice_r+1]-0.5)
                        pub_cmd_vel.publish(msg_cmd_vel)
                    arm=1
                    while(arm==1):
                        hand_palm_link(1,rock_tf[indice_r],rock_tf[indice_r+1],rock_tf[indice_r+2])
                        
                    rospy.sleep(10)
                    #hand_palm_link(rock_tf[indice_r],rock_tf[indice_r+1],rock_tf[indice_r+2])
                    #rospy.sleep(10)
                    grip.close()
                    arm.set_named_target('go')
                    arm.go()
                    indice_r=indice_r+3
                    x_goal,y_goal,theta_Goal = 2.14 - 0.5 - 0.75 , 0.73 , 0
                    aux = aux + 1
                    if (aux == num_rock - 1):
                        aux = 0
                else:    '''
                next_point=next_point+1
                if next_point==5: #4
                    return 'outcome2'
                        #while(1):
                        #move_base(0,0,0,1)
        else:
            if ((angle_to_goal1>=0 and angle_to_goal1<np.pi) or (angle_to_goal1<-np.pi)):
                if abs(angle_to_goal1)>0.1: move_base(0.0,0.0,0.12*np.pi,0.1)                 
                if abs(angle_to_goal1)<0.1: move_base(0.3,0,0,0.1)
            elif ((angle_to_goal1>=np.pi and angle_to_goal1<2*np.pi) or (angle_to_goal1<0 and angle_to_goal1>-(np.pi))):
                if abs(angle_to_goal1)>0.1: move_base(0.0,0.0,-0.12*np.pi,0.1)  
                if abs(angle_to_goal1)<0.1: move_base(0.3,0,0,0.1)
        return 'outcome1'

def main():
    EQUIPO="UPS Cuenca"
    print("Meta Competencia Etapa 5 - " + EQUIPO)
    print('Inicializando')
    r=rospy.Rate(4)	
    rospy.sleep(1)
    punto_ini=get_coords()
    #print ('Tiempo Inicial = '+ str(punto_ini.header.stamp.to_sec()))
    head.go(np.array((0,-.15*np.pi)))
    arm.set_named_target('go')
    arm.go()
    
    
    
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
    
