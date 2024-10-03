#!/usr/bin/env python3


from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2 as cv2
import sys as sys
import numpy as np

class My_App(QtWidgets.QMainWindow):

    def __init__(self):
        #initiates an instance of the QMainWindow class
        super(My_App, self).__init__()
        #loads the user interfance from sift app.ui and applies it to the current instance of the My_App class, called self. 
        loadUi("./SIFT_app.ui", self)

        #some params for the camera
        self._cam_id = 0
        self._cam_fps = 30
        self._is_cam_enabled = False
        self._is_template_loaded = False

        #read inputs from the buttons in the UI
        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        self.template_img = cv2.imread("snowLeopard.jpg", cv2.IMREAD_GRAYSCALE)

        #create a sift object detector
        self.sift = cv2.SIFT_create()
        #detect keypoints and descriptors for the template image
        self.template_keypoints, self.template_descriptors = self.sift.detectAndCompute(self.template_img, None)
        #Use brute force matcher object, which is a method of matching keypoints between two images
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        #initialize camera device
        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

        # Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(1000 / self._cam_fps)

        #this works like an interrupt and checks button inputs. This calls the borse button input method
        self.browse_button.clicked.connect(self.SLOT_browse_button)

    def SLOT_browse_button(self):
        #opens file dialog to allow file selection
        dlg = QtWidgets.QFileDialog()
        #allow selection of file
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]
        #displays images using QPixmap 
        pixmap = QtGui.QPixmap(self.template_path)
        self.template_label.setPixmap(pixmap)
        print("Loaded template image file: " + self.template_path)

    # Source: stackoverflow.com/questions/34232632/
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, 
                        bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)
    
    #The query camera method is used to query the camera and detect the object in the camera frame
    def SLOT_query_camera(self):
        # ret = true or false, if frame was successfully captured 
        # frame = the frame that was captured
        ret, frame = self._camera_device.read()
        
        if ret:
            # gray scale image to make it easy to process
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #compute the descriptors using the grayscale image
            keypoints_frame, descriptors_frame = self.sift.detectAndCompute(gray_frame, None)

            if descriptors_frame is not None:
                #match the descriptors found in the image from the camera feed with the descriptors in the template image
                matches = self.matcher.match(self.template_descriptors, descriptors_frame)

                # Sort matches based on the quality of matches. Shorter distance = higher quality matches
                matches = sorted(matches, key=lambda x: x.distance)

                # Homography calculation
                if len(matches) > 10:
                    # If there are at least 10 matches, extract the coordinates in the images of the matches. 
                    src_pts = np.float32([self.template_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                    # Use RANSAC to compute the homopgraphy matrix. This gives the transformation between the template and the frame. 
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    #This draws the blue square around the detected object
                    if M is not None:
                        # Get the corners of the template image
                        h, w = self.template_img.shape
                        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

                        # Project the template corners to the camera frame
                        dst = cv2.perspectiveTransform(pts, M)

                        # Draw a square around the detected object
                        frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
                    else:
                        # If homography fails, draw lines to the keypoints
                        for m in matches[:10]:
                            pt = keypoints_frame[m.trainIdx].pt
                            cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)

        # Convert the processed frame to QPixmap and display it
        pixmap = self.convert_cv_to_pixmap(frame)
        self.live_image_label.setPixmap(pixmap)

    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")   

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())
