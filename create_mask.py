import numpy as np
import cv2
import glob, os
import scipy.misc


os.chdir("Annotations")

mask_dir = 'Masks'
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

for file in glob.glob("*.png"):
    print(file)

    # Load an color image in grayscale
    img = cv2.imread(file,0)
    ret, mask = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    morph_dialate = 1
    morph_erode = 1
    morph_it = 1
    morph_loop_it = 100
    kernel_dialate = np.ones((morph_dialate, morph_dialate), np.uint8)
    kernel_erode = np.ones((morph_erode, morph_erode), np.uint8)

    for it in range(morph_loop_it):
        mask = cv2.dilate(mask, kernel_dialate, iterations=morph_it)
        mask = cv2.erode(mask, kernel_erode, iterations=morph_it)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    sizes = stats[:, -1]

    img2 = mask
    max_label = 1
    max_size = 30
    for i in range(2, nb_components):
        if sizes[i] < max_size:
            img2[output == i] = 0

    for it in range(morph_loop_it):
        img2 = cv2.dilate(img2, kernel_dialate, iterations=morph_it)
        img2 = cv2.erode(img2, kernel_erode, iterations=morph_it)

    morph_dialate = 5
    morph_erode = 5
    kernel_dialate = np.ones((morph_dialate, morph_dialate), np.uint8)
    img2 = cv2.dilate(img2, kernel_dialate, iterations=morph_it)
    img2 = cv2.erode(img2, kernel_erode, iterations=morph_it)
    scipy.misc.imsave('{}/{}'.format(mask_dir, file), img2)