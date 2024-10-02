import cv2 as cv
import numpy as np

img = cv.imread("snowLeopard.jpg",cv.IMREAD_GREYSCALE)

cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()

    cv.imshow("Image",img)
    cv.imshow("Frame",frame)

    break

