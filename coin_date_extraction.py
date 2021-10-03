from typing import List

from scipy import ndimage as ndi
import numpy as np
import matplotlib.image as mpimg
from skimage.filters import threshold_otsu
from skimage.measure._regionprops import RegionProperties
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, convex_hull_image, watershed
from skimage.color import label2rgb, rgb2gray
from skimage.feature import peak_local_max
from skimage import morphology
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage import exposure
from scipy.stats import mode
from collections import Counter
from skimage import measure
import matplotlib.pyplot as plt
from scipy import ndimage

# Read Images
img1 = mpimg.imread('pic1.jpg')
img = rgb2gray(img1)

elevation_map = sobel(img)
markers = np.zeros_like(img)
markers[img < 140/256] = 1
markers[img > 235/256] = 2
plt.title('Markers')
plt.imshow(markers)
plt.savefig('markers.png')
plt.close()

plt.title('elevation_map')
plt.imshow(elevation_map)
plt.savefig('elevation_map.png')
plt.close()

segmentation = watershed(elevation_map, markers)

plt.title('segmentation')
plt.imshow(segmentation)
plt.savefig('segmentation.png')
plt.close()

segmentation2 = ndi.binary_fill_holes(segmentation - 1)
plt.title('segmentation2')
plt.imshow(segmentation2)
plt.savefig('segmentation2.png')
plt.close()

labeled_coins, _ = ndi.label(segmentation)
plt.title('labeled_coins')
plt.imshow(labeled_coins)
plt.savefig('labeled_coins.png')
plt.close()


def process_all_coins_image(img):
    img = rgb2gray(img)
    elevation_map = sobel(img)
    markers = np.zeros_like(img)
    markers[img < 140/256] = 1
    markers[img > 235/256] = 2
    plt.title('Markers')
    plt.imshow(markers)
    plt.savefig('markers.png')
    plt.close()

    plt.title('elevation_map')
    plt.imshow(elevation_map)
    plt.savefig('elevation_map.png')
    plt.close()

    segmentation = watershed(elevation_map, markers)

    plt.title('segmentation')
    plt.imshow(segmentation)
    plt.savefig('segmentation.png')
    plt.close()

    segmentation2 = ndi.binary_fill_holes(segmentation - 1)
    plt.title('segmentation2')
    plt.imshow(segmentation2)
    plt.savefig('segmentation2.png')
    plt.close()

    labeled_coins, _ = ndi.label(segmentation)
    plt.title('labeled_coins')
    plt.imshow(labeled_coins)
    plt.savefig('labeled_coins.png')
    plt.close()

    return segmentation







image = img

thresh = threshold_otsu(image)


bw = closing(image < thresh, square(3))

# remove artifacts connected to image border
cleared = clear_border(bw, 50)

# label image regions
label_image = label(cleared)
# to make the background transparent, pass the value of `bg_label`,
# and leave `bg_color` as `None` and `kind` as `overlay`
image_label_overlay = label2rgb(label_image, image=image, bg_label=0)


# Get k centroids from k-means wrt the colors
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
image_label_overlay_flattened = image_label_overlay.reshape((-1, 3))

segmentation = ndi.binary_fill_holes(segmentation - 1)
labeled_coins, _ = ndi.label(segmentation)


def mask_to_coins(image):
    image = test_image
    distance = ndimage.distance_transform_edt(image)
    local_maxi = peak_local_max(
        distance, min_distance=150, indices=False, footprint=np.ones((3, 3)), labels=image)
    markers = measure.label(local_maxi)
    labels_ws = watershed(-distance, markers, mask=image)

    markers[~image] = -1
    return labels_ws


# limit = 1e2
limit = 1e6
indicies = [i for i in range(len(set(label_image.flatten()))) if np.sum(label_image[label_image == i]) > limit]

all_labels = []
for _, i in enumerate(indicies):
    test_image = np.zeros_like(label_image)
    test_image[label_image == i] = 1
    tmp_label = mask_to_coins(test_image)
    all_labels.append(tmp_label)

def coin_labels_to_regions(all_labels: List[np.ndarray]) -> List[List[RegionProperties]]:
    """
    Takes a list of numpy arrays with the shape of the image containing all coins. Each numpy array contains segmented
    coins.
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

        output_coin_regions_temp = regionprops(convex_hull_image_1, img)
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


output_coin_regions = coin_labels_to_regions(all_labels)
coin_pics = get_individual_coin_pics(output_coin_regions)

limit = 1e6
img = coin_pics[0]
plt.figure()
plt.title('orig')
plt.imshow(img)

# Equalization
img = exposure.equalize_hist(img)
img = -img
plt.figure()
plt.title('equalize_hist')
plt.imshow(img)



footprint = morphology.disk(3)
white_tophat = morphology.white_tophat(img, footprint)
plt.figure()
plt.title('white tophat')
plt.imshow(img)
contours = measure.find_contours(img, fully_connected='high')


# Display the image and plot all contours found
for contour in contours:
    if len(contour) < 100 and len(contour) > 30:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
    # plt.plot(contour[:, 1], contour[:, 0], linewidth=2)

plt.figure()
plt.hist([len(con) for con in contours if len(con) < 300], 50)
plt.semilogy()
plt.xlim(0, 300)


for img in coin_pics:
    # img = coin_pics[0]
    # plt.figure()
    # plt.title('orig')
    # plt.imshow(img)

    # Equalization
    img = exposure.equalize_hist(img)
    img = -img
    # plt.figure()
    # plt.title('equalize_hist')
    # plt.imshow(img)


    footprint = morphology.disk(3)
    white_tophat = morphology.white_tophat(img, footprint)
    plt.figure()
    plt.title('white tophat')
    plt.imshow(img)
    contours = measure.find_contours(img, fully_connected='high')


    # Display the image and plot all contours found
    all_regionprops = []
    for idx, contour in enumerate(contours):

        if len(contour) < 100 and len(contour) > 30:
            contour_mask = np.zeros_like(img)
            contour_mask[contour.astype(int)[:, 0], contour.astype(int)[:, 1]] = 1
            image_convex_hull = convex_hull_image(contour_mask)
            all_regionprops.append(regionprops(image_convex_hull.astype('uint8')))
            if 50 < all_regionprops[-1][0].area < 200:
                plt.plot(contour[:, 1], contour[:, 0], linewidth=2)

        # plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
        # if idx == 10:
        #     break

    # plt.figure()
    # plt.hist([len(con) for con in contours if len(con) < 300], 50)
    # plt.semilogy()
    # plt.xlim(0, 300)
print(5)
