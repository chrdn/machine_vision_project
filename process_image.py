import matplotlib.pyplot as plt
import numpy as np
import cv2

"""
- image.jpg
- climbing-the-matterhorn-zermatt.jpg
- jet.jpeg
- torii.jpg
"""

sample_image = cv2.imread("jet.jpeg")


# constraints:
# - 256, 3
# - 126, 2
# - 256 * 3//4 , 2

fig, ((ax0, ax1), (ax_b, ax2)) = plt.subplots(2, 2)

img_size = 256 * 3 // 4
K = 3

height, width, channels = sample_image.shape

img = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

# ax0.axis("off")
# ax0.imshow(img)


# img = cv2.resize(img, (img_size, img_size * height // width))
img = cv2.GaussianBlur(img, (5, 5), 0)


ax0.axis("off")

ax0.imshow(img)

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

ax1.axis("off")
ax1.imshow(result_image)


t_lower = 50
t_upper = 150
edge = cv2.Canny(result_image, t_lower, t_upper)


ax2.axis("off")
ax2.imshow(edge)

ax_b.axis("off")

plt.show()
