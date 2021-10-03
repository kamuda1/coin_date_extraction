import matplotlib.image as mpimg
from skimage.color import rgb2gray


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




for img in coin_pics:
    # Equalization
    contours = get_contours(img)
    regionprops_refined = refine_contours(contours, img.shape)
    print(5)
