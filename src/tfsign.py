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
from std_msgs.msg import Int32, Float32

signnames_csv = pd.read_csv("signnames.csv", sep=",")
signnames = list(signnames_csv["SignName"])

tf_model = None

IMG_SIZE = 48
camera_topic_name = "camera"
frame = None
bridge = CvBridge()

sign_idx = -99
sign_prob = -99


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
    # add null dimension to match tf_model.predict()'s 4 dimensional input
    img = np.expand_dims(img, axis=0)
    return img


def classify(img):
    global tf_model, sign_idx, sign_prob
    img = preprocess_img(img)
    if img.shape[1] == 0 or img.shape[2] == 0:
        return

    # YOUR CODE: Predict the class of the given image
    # print("img shape:")
    # print(img.shape)
    # print("tf_model summary:")
    # print(tf_model.summary())
    
    # print("img contents:")
    # for i in range(len(img)):
    #     print(img[i])
    softmax = tf_model.predict(img)
    y = tf_model.predict_classes(img)

    sign_idx = int(y)
    sign_prob = softmax[0, int(y)]

    # YOUR CODE: Print 1) the class number, 2) the name of the traffic sign, and 3) the probability
    # NOTE: 'signnames' is a dictionary that maps tfsign class to name (signnames[int(y)])
    # print("Class Number:", int(y))
    # print("Traffic Sign:", signnames[int(y)])
    # print("Probability:", softmax[0, int(y)])
    print(str(sign_idx) + "\t" + str(sign_prob) + "\t" + signnames[sign_idx])


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

            if p1[0] < 0 or p1[1] < 0 or p2[0] < 0 or p2[1] < 0:
                continue

            cv2.rectangle(img, p1, p2, (255, 0, 255), 3)

            # YOUR CODE: call 'classify' function, passing the image within the bounding box
            #print((p1[1], p2[1]), (p1[0], p2[0]))
            img_roi = img[p1[1] : p2[1], p1[0] : p2[0]]
            classify(img_roi)

    cv2.imshow("HoughCircle", img)
    cv2.waitKey(1)
    print("-----------------------")


if __name__ == "__main__":
    try:
        # Load a trained model
        global tf_model, sign_idx, sign_prob
        tf_model = load_model("tf_model.h5")
        tf_model._make_predict_function()

        rospy.Subscriber(camera_topic_name, Image, image_callback)
        rospy.init_node("tf_sign", anonymous=False)

        sign_idx_pub = rospy.Publisher("sign_idx", Int32, queue_size=1)
        sign_prob_pub = rospy.Publisher("sign_prob", Float32, queue_size=1)

        rate = rospy.Rate(hz=10)
        while not rospy.is_shutdown():
            if frame is not None:
                tf_detect(frame)

                sign_idx_pub.publish(sign_idx)
                sign_prob_pub.publish(sign_prob)
            else:
                print("No frame. Is camera_feeder running?")

            rate.sleep()

    except rospy.ROSInterruptException:
        pass
