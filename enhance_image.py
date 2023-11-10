import matplotlib
matplotlib.use('Agg')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import io

def histogram_equalization(image):
    if len(image.shape) == 2:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist / (image.shape[0] * image.shape[1])
        
        cdf = np.cumsum(hist)
        cdf = cdf * 255

        equalized_image = cdf[image].astype(np.uint8)
    else:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        
        hist = cv2.calcHist([v], [0], None, [256], [0, 256])
        hist = hist / (v.shape[0] * v.shape[1])

        cdf = np.cumsum(hist)
        cdf = cdf * 255
        
        v = cdf[v].astype(np.uint8)
        equalized_image = cv2.merge([h, s, v])
        equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_HSV2BGR)

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.title("Histogram of Initial Image")
    plt.hist(image.flatten(), 256, [0, 256], color='r')

    plt.subplot(2, 2, 2)
    plt.title("Histogram of Equalized Image")
    plt.hist(equalized_image.flatten(), 256, [0, 256], color='r')

    plt.subplot(2, 2, 3)
    plt.title("Initial Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Equalized Image")
    plt.imshow(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    merged_image = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), 1)
    plt.close()

    return merged_image
