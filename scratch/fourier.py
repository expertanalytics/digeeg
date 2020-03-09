import cv2

import numpy as np
import matplotlib.pyplot as plt


def color_to_grayscale(color_array: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(color_array, cv2.COLOR_BGR2GRAY)


def calcPSD(input_image, output_image, flag):
    # Complex input image with zero imaginary component

    X = np.fft.rfft2(input_image).real
    np.power(X, 2, out=X)
    # X += 1
    # np.log(X, out=X)

    # # plt.imshow(X, cmap="gray")
    # plt.imshow(X)
    # plt.show()
    print(X)


    complex_image = np.zeros(shape=(input_image.shape[0], input_image.shape[1], 2), dtype=np.float32)
    complex_image[:, :, 0] = input_image


    cv2.dft(complex_image, dst=complex_image)
    complex_image[:, 0, :] = 0      # Hmm
    psd = cv2.magnitude(complex_image[:, :, 0], complex_image[:, :, 1])
    cv2.pow(psd, 2, dst=psd)
    print(psd)

    if flag:
        imlog = psd + 1
        np.log(imlog, out=imlog)
        output_image = imlog
    else:
        output_image = psd

    # print(output_image)
    plt.imshow(X, cmap="gray")
    plt.show()



if __name__ == "__main__":
    image = cv2.imread("test4_split0.png")
    # image = cv2.imread("test.png")
    image = color_to_grayscale(image)
    print("image type: ", image.dtype)

    new_size = image.shape[0] & -2, image.shape[1] & -2     # Even number of rows / columns
    new_image = np.zeros(shape=new_size, dtype=np.float32)
    new_image[:, :] = image[:new_size[0], :new_size[1]]

    calcPSD(new_image, new_size, 0)
    print("Success!")
