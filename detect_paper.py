"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""

import sys
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


sample_image = cv2.imread("paper_on_table.jpeg")

img = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img, 50, 200, None, 3)


lines = cv2.HoughLines(edges, 1, np.pi / 180, 175)

for r_theta in lines:
    arr = np.array(r_theta[0], dtype=np.float64)
    r, theta = arr
    # Stores the value of cos(theta) in a
    a = np.cos(theta)

    # Stores the value of sin(theta) in b
    b = np.sin(theta)

    # x0 stores the value rcos(theta)
    x0 = a * r

    # y0 stores the value rsin(theta)
    y0 = b * r

    # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
    x1 = int(x0 + 1000 * (-b))

    # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
    y1 = int(y0 + 1000 * (a))

    # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
    x2 = int(x0 - 1000 * (-b))

    # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
    y2 = int(y0 - 1000 * (a))

    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
    # (0,0,255) denotes the colour of the line to be
    # drawn. In this case, it is red.
    cv2.line(sample_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

plt.axis("off")
plt.imshow(sample_image)
plt.show()
