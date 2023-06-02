#!/usr/bin/ python3

import queue
import cv2
import rospy
import torch
import numpy as np
import ros_numpy

from super_gradients.training import models

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2
from mot import yolonas_deepSORT


class Detector:
    def __init__(self, ros_rate):
        self.loadParameters()
        self.bridge = CvBridge()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model = models.get("yolo_nas_s", pretrained_weights="coco").to(self.device)
        self.rgb_image = None
        self.tracker = yolonas_deepSORT("/home/dorleco/yolo-nas/deep_sort/model_weights/mars-small128.pb", self.model)
        self.ros_rate = ros_rate
        self.depth_image = None
        self.resized_depth_image = None
        self.detection_boxes = []
        self.detected_class_names = []

        self.RGB_IMAGE_RECEIVED = 0
        self.DEPTH_IMAGE_RECEIVED = 0

        self.custom_classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 
                            #    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'airplane'
                               ]

    
    def loadParameters(self):
        self.image_topicname = rospy.get_param(
            "object_detection/image_topic_name", "/carla/ego_vehicle/rgb_front/image")
        self.depth_topicname = rospy.get_param(
            "object_detection/depth_topic_name", "/carla/ego_vehicle/depth_front/image")
        self.lidar_topicname = rospy.get_param(
            "object_detection/lidar_topic_name", "/carla/ego_vehicle/lidar")
        self.detect_pub_topic_name = rospy.get_param(
            "object_detection/detect_pub_image_topic_name", "/object/tracked_image")
        self.depth_pub_topic_name = rospy.get_param(
            "object_detection/depth_pub_image_topic_name", "/object/depth_image")


    def subscribeToTopics(self):
        rospy.loginfo("Subscribed to topics")
        rospy.Subscriber(self.image_topicname, Image,
                         self.storeImage, buff_size = 2**24, queue_size=1)
        rospy.Subscriber(self.depth_topicname, Image,
                         self.storeDepth, buff_size = 2**24, queue_size=1)
        rospy.Subscriber(self.lidar_topicname, PointCloud2,
                         self.storeLidar, buff_size = 2**24, queue_size=1)

    
    def publishToTopics(self):
        rospy.loginfo("Published to topics")
        self.detectionPublisher = rospy.Publisher(self.detect_pub_topic_name, Image, queue_size=1)
        self.depthPublisher = rospy.Publisher(self.depth_pub_topic_name, Image, queue_size=1)


    
    def storeImage(self, img): # Copy for Obj Detection
        try:
            frame = self.bridge.imgmsg_to_cv2(img, 'bgr8')
            # rospy.loginfo("RGB Image Stored")
        except CvBridgeError as e:
            rospy.loginfo(str(e))
        self.rgb_image = frame
        self.RGB_IMAGE_RECEIVED = 1
        self.track()
        self.sync_frames()

    def storeDepth(self, img):
        frame = None
        try:
            frame = self.bridge.imgmsg_to_cv2(img, '32FC1')
            # rospy.loginfo("RGB Image Stored")
        except CvBridgeError as e:
            rospy.loginfo(str(e))
        self.depth_image = frame
        self.DEPTH_IMAGE_RECEIVED = 1
        print("Published")
        self.callPublisherDepth(self.depth_image)
        self.sync_frames()

    def storeLidar(self, pc):
        # print("Storing lidar")
        frame = None
        try:
            xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pc)
            # print(xyz_array)
        except CvBridgeError as e:
            rospy.loginfo(str(e))

        

    def track(self):
        # print(np.array(self.rgb_image)[:,:,0:3].shape)
        detected_img, self.detection_boxes, self.detected_class_names = self.tracker.track(np.array(self.rgb_image)[:,:,0:3])
        print("Published")
        self.callPublisher(detected_img)

    def sync_frames(self):
        if self.RGB_IMAGE_RECEIVED == 1 and self.DEPTH_IMAGE_RECEIVED == 1:
            # print(self.detection_boxes)
            # print()
            # print("---------------------------------------------")
            # print(self.detected_class_names)
            # print()
            # print("---------------------------------------------")
            # print(len(self.detected_class_names))
            # print(len(self.detection_boxes))
            self.resized_depth_image = cv2.resize(self.depth_image, (600, 800), interpolation=cv2.INTER_AREA)
            print(self.depth_image.shape)
            print(self.rgb_image.shape)
            centers = []
            names = []
            for i in range(len(self.detected_class_names)):
                if self.detected_class_names[i] in self.custom_classes:
                    x_center = (self.detection_boxes[i][0] + self.detection_boxes[i][2])//2
                    y_center = (self.detection_boxes[i][1] + self.detection_boxes[i][3])//2
                    centers.append([x_center, y_center])
                    names.append(self.detected_class_names[i])

            distances = self.calculateDistance(centers)
            
            if len(centers) != 0:
                idx = np.argmin(distances)
                print(distances[idx], names[idx])



    def callPublisher(self, image):
        detected_img = self.bridge.cv2_to_imgmsg(image, 'bgr8')
        self.detectionPublisher.publish(detected_img)

    def callPublisherDepth(self, image):
        depth_img = self.bridge.cv2_to_imgmsg(image, '32FC1')
        self.depthPublisher.publish(depth_img)

    def calculateDistance(self, centers):
        distances = []
        for center in centers:
            print(center[0], center[1])
            dist = self.resized_depth_image[int(center[0])][int(center[1])]
            distances.append(dist)

        return distances