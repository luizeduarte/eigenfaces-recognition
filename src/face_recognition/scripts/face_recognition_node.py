#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from face_msgs.msg import Facebox, Faces
from cv2 import CascadeClassifier
from cv2.face import EigenFaceRecognizer

class face_recognition_node(Node):
    def __init__(self):
        super().__init__('face_recognition_node')	# Initialize the node	
        self.bridge = CvBridge()	# For converting ROS Image messages to OpenCV images

        # Load the face detection and recognition models
        self.face_cascade = CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.EigenFaceRecognizer_create()
        self.declare_parameter('model_path', 'models/eigen_model.xml')
        self.recognizer.read(self.get_parameter('model_path').value)
        
	# Create a subscriber to receive the camera images
        self.subscription = self.create_subscription(Image, '/webcam_image',  self.image_callback, 10)
        
	# Create a publisher to publish the detected faces
        self.publisher = self.create_publisher(Faces, '/faces', 5)

    def detect_faces(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)	# Convert the frame to grayscale
        faces_coordinates = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces_coordinates

    def image_callback(self, msg):
        # Convert the ROS Image message to an OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        faces_list = self.detect_faces(cv_image)	# Detect faces in the image

        faces_msg = Faces()				# Create a message to publish the detected faces
        faces_msg.image_width = cv_image.shape[1]
        faces_msg.image_height = cv_image.shape[0]

        for (x, y, w, h) in faces_list:
            # Extract the face region of interest (ROI)
            face_roi = cv_image[y:y + h, x:x + w]
            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # resize the face to the size the recognizer expects
            face_gray = cv2.resize(face_gray, (100, 100))

            # Recognize the face using the trained recognizer
            label, confidence = self.recognizer.predict(face_gray)

            face_box = Facebox()
            face_box.top = int(y)
            face_box.left = int(x)
            face_box.width = int(w)
            face_box.height = int(h)
            face_box.label = label
            face_box.confidence = confidence
            faces_msg.face_boxes.append(face_box)

            # Draw a rectangle around the detected face
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (100, 100, 255), 1)

            # draw confidence
            cv2.putText(cv_image, f'{confidence:.2f}', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
	    # draw label
            cv2.putText(cv_image, f'{label}', (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Draw "recognized" if the confidence is high enough
            if confidence < 4100:
                cv2.putText(cv_image, "Recognized", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(cv_image, "Unrecognized", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


	# Publish the detected faces
        self.publisher.publish(faces_msg)

        cv2.imshow("Live Feed", cv_image)
        cv2.waitKey(3)

if __name__ == '__main__':
        rclpy.init()
        face_recognition_node = face_recognition_node()
        rclpy.spin(face_recognition_node)
        rclpy.shutdown()

