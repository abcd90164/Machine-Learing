import cv2
import numpy as np
from keras.datasets import mnist
from keras import utils

SHIFT_UNIT = 5

# Function to shift image data left
def shift_left(matrix, num_pixels):
    """Shift matrix elements to the left and fill the emptied part with zeros."""
    new_matrix = np.zeros_like(matrix)
    new_matrix[:, :-num_pixels] = matrix[:, num_pixels:]
    return new_matrix

def shift_right(matrix, num_pixels):
    """Shift matrix elements to the right and fill the emptied part with zeros."""
    new_matrix = np.zeros_like(matrix)
    new_matrix[:, num_pixels:] = matrix[:, :-num_pixels]
    return new_matrix

def shift_up(matrix, num_pixels):
    """Shift matrix elements upward and fill the emptied part with zeros."""
    new_matrix = np.zeros_like(matrix)
    new_matrix[:-num_pixels, :] = matrix[num_pixels:, :]
    return new_matrix

def shift_down(matrix, num_pixels):
    """Shift matrix elements downward and fill the emptied part with zeros."""
    new_matrix = np.zeros_like(matrix)
    new_matrix[num_pixels:, :] = matrix[:-num_pixels, :]
    return new_matrix

def display_image_with_opencv(image):
    """Display an MNIST image using OpenCV."""
    # cv2.imshow('MNIST Image', image)
    # cv2.waitKey(0)  # Wait for a key press to close the window
    # cv2.destroyAllWindows()

# Load MNIST data
(x_train_src, y_train_src), (x_test, y_test) = mnist.load_data()

# Select one image
image = x_train_src[0]
display_image_with_opencv(image)

# Apply the shifts to each image
x_train_shifted_left = np.array([shift_left(img, SHIFT_UNIT) for img in x_train_src])
x_train_shifted_right = np.array([shift_right(img, SHIFT_UNIT) for img in x_train_src])
x_train_shifted_up = np.array([shift_up(img, SHIFT_UNIT) for img in x_train_src])
x_train_shifted_down = np.array([shift_down(img, SHIFT_UNIT) for img in x_train_src])

# Select one image
image = x_train_shifted_left[0]
display_image_with_opencv(image)
image = x_train_shifted_right[0]
display_image_with_opencv(image)
image = x_train_shifted_up[0]
display_image_with_opencv(image)
image = x_train_shifted_down[0]
display_image_with_opencv(image)

# Combine all datasets
x_train_combined = np.vstack([x_train_src, x_train_shifted_left, x_train_shifted_right, x_train_shifted_up, x_train_shifted_down])
y_train_combined = np.repeat(y_train_src, 5)  # Replicate labels for each shifted version

# Optionally, you could also shuffle the dataset here if needed
# print("Combined dataset shape:", x_train_combined.shape)
# print("Labels shape:", y_train_combined.shape)


# 資料預處理
x_train = x_train_combined
y_train = y_train_combined

# 訓練集資料
x_train = x_train.reshape(x_train.shape[0],-1)  # 轉換資料形狀
x_train = x_train.astype('float32')/255         # 轉換資料型別
y_train = y_train.astype(np.float32)

# 測試集資料
x_test = x_test.reshape(x_test.shape[0],-1)     # 轉換資料形狀
x_test = x_test.astype('float32')/255           # 轉換資料型別
y_test = y_test.astype(np.float32)

knn=cv2.ml.KNearest_create()                    # 建立 KNN 訓練方法
knn.setDefaultK(5)                             # 參數設定
knn.setIsClassifier(True)

print('training...')
knn.train(x_train, cv2.ml.ROW_SAMPLE, y_train)  # 開始訓練
knn.save('mnist_knn.xml')                       # 儲存訓練模型
print('ok')

print('testing...')
test_pre = knn.predict(x_test)                  # 讀取測試集並進行辨識
test_ret = test_pre[1]
test_ret = test_ret.reshape(-1,)
test_sum = (test_ret == y_test)
acc = test_sum.mean()                           # 得到準確率
print(acc)
