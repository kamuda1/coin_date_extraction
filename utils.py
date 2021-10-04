from typing import List, Tuple

import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure._regionprops import RegionProperties
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, convex_hull_image, watershed, binary_dilation
from skimage.feature import peak_local_max
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage import exposure

from scipy.stats import mode
from collections import Counter
from skimage import measure
import scipy.ndimage as ndimage


def refine_regions(dilated_mask: np.ndarray, img: np.ndarray,  percentile: float = 80):
    regions_labeled = label(dilated_mask)
    regions_labeled_props = regionprops(regions_labeled, img)

    all_major_axis_lengths = [region.major_axis_length for region in regions_labeled_props]
    major_axis_length_thresh = np.percentile(all_major_axis_lengths, percentile)

    best_good_region_candidates = []
    for region in regions_labeled_props:
        if region.major_axis_length > major_axis_length_thresh:
            best_good_region_candidates.append(region)
    return best_good_region_candidates


def regionprops_to_region_mask(regionprops, img_shape: Tuple):
    r_mask = np.zeros(img_shape, dtype='bool')
    for regionprops in regionprops:
        contour = regionprops.coords
        # Create an empty image to store the masked array
        # Create a contour image by using the contour coordinates rounded to their nearest integer value
        r_mask[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
        # Fill in the hole created by the contour boundary
        r_mask += ndimage.binary_fill_holes(r_mask)

    return r_mask


def dilate_and_join_regions(all_regionprops: List[RegionProperties], img_shape: Tuple, mask_size: int = 5):
    mask = regionprops_to_region_mask(all_regionprops, img_shape)
    dilated_mask = binary_dilation(mask, square(mask_size))
    return dilated_mask


def refine_contours(contours: List[np.ndarray], img_shape: Tuple) -> List[RegionProperties]:
    """
    Refine contours and return the regions that fit a threshold.
    """
    all_regionprops = []
    for contour in contours:
        if len(contour) < 100 and len(contour) > 30:
            contour_mask = np.zeros(img_shape)
            contour_mask[contour.astype(int)[:, 0], contour.astype(int)[:, 1]] = 1
            image_convex_hull = convex_hull_image(contour_mask)
            region_props_tmp = regionprops(image_convex_hull.astype('uint8'))[0]
            if 50 < region_props_tmp.area < 200:
                all_regionprops.append(region_props_tmp)
    return all_regionprops


def get_contours(img: np.ndarray) -> List[np.ndarray]:
    """
    Finds contours from a grayscale image. Output is a list of contours. Each item in
    the list is an num_contours x 2 array defining the row, col values of pixels on
    contours.
    """
    img_equalized = -exposure.equalize_hist(img)
    contours = measure.find_contours(img_equalized, fully_connected='high')
    return contours


def get_largest_regions(label_image: np.ndarray, count_thresh: float = 1e6) -> List[np.ndarray]:
    """
    Gets regions with a population above count_thresh, an empirically derived value is set as default.
    """

    indicies = [i for i in range(len(set(label_image.flatten())))
                if np.sum(label_image[label_image == i]) > count_thresh]

    large_region_labels = []
    for _, i in enumerate(indicies):
        test_image = np.zeros_like(label_image)
        test_image[label_image == i] = 1
        tmp_label = mask_to_coins(test_image)
        large_region_labels.append(tmp_label)

    return large_region_labels


def get_all_coins_overlay(image: np.ndarray) -> np.ndarray:
    """
    image is a 2d grayscale image with values in (0, 1). label_image is an image with the same shape as the input image,
    but with each connecting region labeled with an integer. Coarse, output is many small and some large regions.
    """
    thresh = threshold_otsu(image)
    coins_mask = closing(image < thresh, square(3))

    # remove artifacts connected to image border. Doesn't do much?
    cleared = clear_border(coins_mask, 50)

    # label image regions
    label_image = label(cleared)

    return label_image


def coin_labels_to_regions(all_labels: List[np.ndarray], all_coins_image: np.ndarray) -> List[List[RegionProperties]]:
    """
    Takes a list of numpy arrays with the shape of the image containing all coins. Each numpy array contains segmented
    coins. all_coins_image is a grayscale image with channels in the last index.
    """
    output_coin_regions = []
    for test_img in all_labels:
        ctr = Counter(test_img.ravel())
        second_most_common_value, its_frequency = ctr.most_common(2)[1]

        cluster_index = mode(test_img)
        test_img_result = np.zeros_like(test_img)
        test_img_result[test_img == second_most_common_value] = 1

        convex_hull_coin_mask = convex_hull_image(test_img_result)
        convex_hull_image_1 = np.zeros_like(convex_hull_coin_mask, dtype='int')
        convex_hull_image_1[convex_hull_coin_mask == True] = 1

        output_coin_regions_temp = regionprops(convex_hull_image_1, all_coins_image)
        output_coin_regions.append(output_coin_regions_temp)
    return output_coin_regions


def get_individual_coin_pics(coin_regions: List[List[RegionProperties]]) -> List[np.ndarray]:
    """
    Takes a list of RegionProperties objects and extracts the image from each, returning a list of images in 2d numpy
    arrays. coin_regions might have multiple coins in each region.
    """
    coin_pics = []
    for region in coin_regions:
        coin_pics.append(region[0].intensity_image)
    return coin_pics


def process_all_coins_image(img):
    """
    First pass segmentation. Segments images using a sobel edge detector and rough empirical thresholds.

    img must be (channels_x, channels_y, 1) grayscale image.
    """
    elevation_map = sobel(img)
    markers = np.zeros_like(img)
    markers[img < 140/256] = 1
    markers[img > 235/256] = 2
    segmentation = watershed(elevation_map, markers)

    return segmentation

def mask_to_coins(image):
    distance = ndimage.distance_transform_edt(image)
    local_maxi = peak_local_max(
        distance, min_distance=150, indices=False, footprint=np.ones((3, 3)), labels=image)
    markers = measure.label(local_maxi)
    labels_ws = watershed(-distance, markers, mask=image)

    markers[~image] = -1
    return labels_ws
