import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_new(old):
    new = np.ones(old.shape, np.uint8)
    cv2.bitwise_not(new, new)
    return new


def rotate_image(image, angle):
    # DOES NOT CENTER THE IMAGE
    # ANGLE CAN BE FOUND DIRECTLY BY USING CONTOUR END POINTS
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


# - paper_on_table.jpeg
# - "z13MW.jpg"
# - y6Nq8.jpg
# - "lone_paper.png"
# - "lone_paper_slanted.png"

sample_image = cv2.imread("ultimate.jpg")

gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(blurred, 0, 200)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


passing_contours = []

for contour in contours:

    # Approximate the contour

    arc_length = cv2.arcLength(contour, True)

    approx = cv2.approxPolyDP(contour, 0.02 * arc_length, True)
    # If the approximated contour has 4 points, we can consider it a rectangle

    if len(approx) == 4:
        cv2.drawContours(sample_image, [approx], -1, (0, 255, 0), 3)
        passing_contours = passing_contours + [contour]


max_area = 0
max_contour = approx
for passing_contour in passing_contours:
    current_area = cv2.contourArea(passing_contour)
    # print(max_area, current_area)
    if current_area > max_area:
        max_contour = passing_contour
        max_area = current_area

mask = np.zeros_like(sample_image)
cv2.drawContours(sample_image, [max_contour], -1, (0, 255, 0), -1)

cv2.drawContours(mask, [max_contour], -1, (255, 255, 255), -1)


mask_out = cv2.bitwise_and(sample_image, mask)
alpha = np.sum(mask_out, axis=-1) > 0
alpha = np.uint8(alpha * 255)
mask_out = np.dstack((mask_out, alpha))

rotated = rotate_image(mask_out, 180)

plt.axis("off")
plt.imshow(sample_image)
plt.show()
