import matplotlib.pyplot as plt
import numpy as np
import cv2
import detect_paper

"""
- image.jpg
- climbing-the-matterhorn-zermatt.jpg
- jet.jpeg
- torii.jpg
"""


def process_image(sample_image):

    # constraints:
    # - 256, 3
    # - 126, 2
    # - 256 * 3//4 , 2

    return_list = []

    img_size = 256 * 3 // 4
    K = 3

    height, width, channels = sample_image.shape

    img = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

    return_list = [detect_paper.alpha_out(img)]

    img = cv2.resize(img, (img_size, img_size * height // width))
    img = cv2.GaussianBlur(img, (5, 5), 0)

    return_list = return_list + [detect_paper.alpha_out(img)]

    twoDimage = img.reshape((-1, 3))
    twoDimage = np.float32(twoDimage)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    attempts = 10

    ret, label, center = cv2.kmeans(
        twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))

    return_list = return_list + [detect_paper.alpha_out(result_image)]

    t_lower = 50
    t_upper = 150
    edge = cv2.Canny(result_image, t_lower, t_upper)

    return_list = return_list + [edge]

    return return_list


if __name__ == "__main__":
    import sys

    sample_image = cv2.imread(sys.argv[1])
    final_images = process_image(sample_image)

    fig, ((ax0, ax1), (ax_b, ax2)) = plt.subplots(2, 2)

    for ax, img in zip([ax0, ax1, ax2, ax_b], final_images):
        ax.axis("off")
        ax.imshow(img)

    plt.show()

# import os

# dirname = os.path.dirname(__file__)
# filename = os.path.join(dirname, "detect_paper_test_image/envelope.jpg")
# sample_image = cv2.imread(filename)
# detect_paper(sample_image)
