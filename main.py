#!/usr/bin/ python3

import rospy
from ros_bridge_cv import Detector

def main():
    print("Called main")
    rospy.loginfo("Called main")
    rospy.init_node('object_tracking')
    rate = rospy.Rate(3)
    detector = Detector(ros_rate = rate)
    detector.subscribeToTopics()
    detector.publishToTopics()
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException as e:
        print(e)
        pass