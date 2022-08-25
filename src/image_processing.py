import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_image(name_file):
    img = cv2.imread(name_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def create_gaussian_noise_mat(h, l):
    n = np.random.normal(0, 1, h*l)
    n = np.reshape(n, (h, l)) * 127
    return n


def add_noise_on_image(img, noise, k):
    img = img.astype('float64')
    for i in range(3):
        img[:, :, i] = img[:, :, i] + k * noise
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype('int32')
    return img


if __name__ == '__main__':
    print("image processing...")
    photo_or = get_image("data/photo_identitee.jpg")
    photo = photo_or.copy()
    h, l, n = photo.shape
    noise = create_gaussian_noise_mat(h, l)
    photo_w_noise = add_noise_on_image(photo, noise, 0.01)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(photo_or)
    plt.subplot(1, 2, 2)
    plt.imshow(photo_w_noise)
    plt.show()

