#!/usr/bin/env python
import numpy as np
import cv2
import math
import sys
from keras.models import load_model
import datetime
import pandas as pd

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int32

signnames_csv = pd.read_csv("signnames.csv", sep=",")
signnames = list(signnames_csv["SignName"])

tf_model = None

IMG_SIZE = 48
camera_topic_name = "camera"
frame = None
bridge = CvBridge()


def image_callback(ros_image):
    global frame
    frame = bridge.imgmsg_to_cv2(ros_image, "bgr8")


# DO NOT MODIFY
def preprocess_img(img):
    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[
        centre[0] - min_side // 2 : centre[0] + min_side // 2,
        centre[1] - min_side // 2 : centre[1] + min_side // 2,
        :,
    ]
    # rescale to standard size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # roll color axis to axis 0
    img = np.rollaxis(img, -1)
    return img


def classify(img):
    img = preprocess_img(img)
    if img.shape[1] == 0 or img.shape[2] == 0:
        return

    # YOUR CODE: Predict the class of the given image
    softmax = model.predict(img)
    y = model.predict_classes(img)

    # YOUR CODE: Print 1) the class number, 2) the name of the traffic sign, and 3) the probability
    # NOTE: 'signnames' is a dictionary that maps tfsign class to name (signnames[int(y)])
    print("Class Number:", int(y))
    print("Traffic Sign:", signnames[int(y)])
    print("Probability:", softmax)


def tf_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    rows = gray.shape[0]

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        1,
        rows / 8,
        param1=100,
        param2=30,
        minRadius=20,
        maxRadius=40,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # center: i[0], i[1]
            # radius: i[2]

            if i[0] < i[2] or i[1] < i[2]:
                continue

            # YOUR CODE: draw the outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (127, 0, 127), 2)

            # YOUR CODE: draw the center of the circle
            cv2.circle(img, (i[0], i[1]), 3, (0, 255, 0), 4)

            # YOUR CODE: draw a bounding box (with some margin)
            margin = 10
            p1 = (i[0] - i[2] - margin, i[1] - i[2] - margin)
            p2 = (i[0] + i[2] + margin, i[1] + i[2] + margin)
            cv2.rectangle(img, p1, p2, (255, 0, 255), 3)

            # YOUR CODE: call 'classify' function, passing the image within the bounding box
            img = img[p1[1] : p2[1], p1[0] : p2[0]]
            # classify(img)

    cv2.imshow("HoughCircle", img)
    cv2.waitKey(1)
    print("-----------------------")


if __name__ == "__main__":
    try:
        # Load a trained model
        global tf_model
        tf_model = load_model("tf_model.h5")
        tf_model._make_predict_function()

        rospy.Subscriber(camera_topic_name, Image, image_callback)
        rospy.init_node("tf_sign", anonymous=False)

        rate = rospy.Rate(hz=10)
        while not rospy.is_shutdown():
            if frame is not None:
                tf_detect(frame)
            else:
                print("No frame. Is camera_feeder running?")

            rate.sleep()

    except rospy.ROSInterruptException:
        pass
