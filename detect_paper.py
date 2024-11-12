import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_paper(sample_image):

    def get_new(old):
        new = np.ones(old.shape, np.uint8)
        cv2.bitwise_not(new, new)
        return new

    def rotate_image(image, angle, center_coord):
        # DOES NOT CENTER THE IMAGE
        # ANGLE CAN BE FOUND DIRECTLY BY USING CONTOUR END POINTS
        h, w, c = image.shape

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        tran_mat = np.float32(
            [[1, 0, w / 2 - center_coord[0]], [0, 1, h / 2 - center_coord[1]]]
        )
        result = cv2.warpAffine(image, tran_mat, image.shape[1::-1])
        result = cv2.warpAffine(
            result, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR
        )
        return result

    def find_angle(list_of_angles):
        min_value = 400
        min_value_abs = 400
        for a in list_of_angles:
            if abs(a) < min_value_abs:
                min_value = a
                min_value_abs = abs(a)
        return min_value

    # - paper_on_table.jpeg
    # - "z13MW.jpg"
    # - y6Nq8.jpg
    # - "lone_paper.png"
    # - "lone_paper_slanted.png"
    # - "envelope"

    gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 0, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    passing_contours = []
    passing_approxs = []

    for contour in contours:

        # Approximate the contour

        arc_length = cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(contour, 0.02 * arc_length, True)
        # If the approximated contour has 4 points, we can consider it a rectangle

        if len(approx) == 4:
            # cv2.drawContours(sample_image, [approx], -1, (0, 255, 0), 3)
            passing_contours = passing_contours + [contour]
            passing_approxs = passing_approxs + [approx]

    # select max contour
    max_area = 0
    max_contour = approx
    max_approx = approx
    for passing_contour, passing_approx in zip(passing_contours, passing_approxs):
        current_area = cv2.contourArea(passing_contour)
        # print(max_area, current_area)
        if current_area > max_area:
            max_contour = passing_contour
            max_area = current_area
            max_approx = passing_approx

    # find corners to calculate angle
    rot_rect = cv2.minAreaRect(max_approx)
    box = cv2.boxPoints(rot_rect)

    l = box
    slope = [[l[i][0] - l[i + 1][0], l[i][1] - l[i + 1][1]] for i in range(len(l) - 1)]
    angles = [np.degrees(np.arctan(s[1] / s[0])) for s in slope]

    mask = np.zeros_like(sample_image)
    # cv2.drawContours(sample_image, [max_contour], -1, (0, 255, 0), -1)

    cv2.drawContours(mask, [max_contour], -1, (255, 255, 255), -1)

    mask_out = cv2.bitwise_and(sample_image, mask)

    M = cv2.moments(max_contour)
    # cv2.circle(mask_out,(round(M["m10"] / M["m00"]), round(M["m01"] / M["m00"])),5,(0, 255, 0),-1,)

    rotated = rotate_image(
        mask_out,
        find_angle(angles),
        [round(M["m10"] / M["m00"]), round(M["m01"] / M["m00"])],
    )

    return rotated

    plt.axis("off")
    plt.imshow(rotated)
    plt.show()


def alpha_out(mask_out):
    alpha = np.sum(mask_out, axis=-1) > 0
    alpha = np.uint8(alpha * 255)
    return np.dstack((mask_out, alpha))
