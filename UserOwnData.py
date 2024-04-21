import cv2
import numpy as np

# Load an image
image = cv2.imread('C:\\Users\\user\\Documents\\AI\\num_9.png', cv2.IMREAD_GRAYSCALE)

cv2.imshow('img', image)
cv2.waitKey()

# Resize and normalize
image = cv2.resize(image, (28, 28))

cv2.imshow('img', image)
cv2.waitKey()

image = image.reshape(-1) / 255.0  # Flatten and normalize

cv2.waitKey()
cv2.destroyAllWindows()


# Assume the label for your image
# label = np.array([your_label]).astype(np.float32)
