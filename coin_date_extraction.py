import matplotlib
# matplotlib.use('Agg')

from utils import regionprops_to_region_mask, dilate_and_join_regions, refine_regions
import matplotlib.image as mpimg
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


from utils import coin_labels_to_regions, get_individual_coin_pics, process_all_coins_image, mask_to_coins, \
    get_all_coins_overlay, get_largest_regions, get_contours, refine_contours

# Read Images
all_coins_pic_rgb = mpimg.imread('pic1.jpg')
all_coins_pic_gray = rgb2gray(all_coins_pic_rgb)

segmentation = process_all_coins_image(all_coins_pic_gray)
label_image = get_all_coins_overlay(all_coins_pic_gray)

large_region_labels = get_largest_regions(label_image)
output_coin_regions = coin_labels_to_regions(large_region_labels, all_coins_pic_gray)
coin_pics = get_individual_coin_pics(output_coin_regions)

import numpy as np
from skimage.transform import rotate
from skimage.filters import sobel, roberts
from skimage.exposure import equalize_hist
from skimage.measure import find_contours

for img in coin_pics:
    contours = get_contours(img)
    regionprops_refined = refine_contours(contours, img.shape)
    dilated_mask = dilate_and_join_regions(regionprops_refined, img.shape, mask_size=8)
    regionprops_refined_2 = refine_regions(dilated_mask, img, percentile=90)
    for region in regionprops_refined_2:
        # plt.figure()
        plt.title(str(region.orientation) + ' ' + str(region.orientation * 180 / np.pi))
        angle = region.orientation * 180 / np.pi
        image_rotated = rotate(region.intensity_image, angle + 90, True)

        img_modified = roberts(equalize_hist(image_rotated))
        img_modified[img_modified < 0.1] = np.average(img_modified)

        contours = find_contours(img_modified, fully_connected='high')
        plt.figure()
        plt.imshow(img_modified)

        plt.figure()
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0])
        # plt.imshow(equalize_hist(image_rotated)) # , square(5)))
        plt.gca().invert_yaxis()
        plt.show()
        # plt.imshow(image_rotated)
    # plt.show()

    print(5)
