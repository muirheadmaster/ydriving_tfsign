#!/usr/bin/env python
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import pyrealsense2 as rs
topic_name_color = 'camera'
topic_name_ir = 'camera_ir'

if __name__ == '__main__':
    camera_color_pub = rospy.Publisher(topic_name_color,Image,queue_size=1)
    camera_ir_pub = rospy.Publisher(topic_name_ir,Image,queue_size=1)
    rospy.init_node('camera_feeder', anonymous=False)
    print('Opening D435 Pipeline')
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) #640x480 image, frequency: 30 Hz
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    pipeline.start(config)
    print('D435 Pipeline starts - publishing `%s` and `%s`' % (topic_name_color, topic_name_ir) )
    bridge = CvBridge()
    while not rospy.is_shutdown():
        try:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            
            ir_frame = frames.get_infrared_frame(1)
            ir_image = np.asanyarray(ir_frame.get_data())
            ros_image = bridge.cv2_to_imgmsg(color_image, "bgr8")
            camera_color_pub.publish(ros_image)
            ros_image = bridge.cv2_to_imgmsg(ir_image, "mono8")
            camera_ir_pub.publish(ros_image)

        except Exception as e:
            print(e)
            break
    pipeline.stop()
    cv2.destroyAllWindows()
