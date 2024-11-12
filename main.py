import sys
import cv2
import matplotlib.pyplot as plt
import detect_paper
import process_image


def main(filename):
    sample_image = cv2.imread(filename)
    oriented_image = detect_paper.detect_paper(sample_image)
    final_images = process_image.process_image(oriented_image)

    fig, ((ax0, ax1), (ax_b, ax2)) = plt.subplots(2, 2)

    for ax, img in zip([ax0, ax1, ax2, ax_b], final_images):
        ax.axis("off")
        ax.imshow(img)

    plt.show()

    plt.figure(2)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1])
