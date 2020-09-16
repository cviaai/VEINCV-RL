import numpy as np
import math
import cv2
# %matplotlib inline
import matplotlib.pyplot as plt
from skimage import data
import scipy.signal
from skimage.feature import match_template
import imutils
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from os import listdir
from os.path import isfile, join
import datetime


def convex_hull(image):
    """ Calculate the convex hull of a image.

    Parameters:
    mask: image (int n*m matrix). grayscale

    Returns:
    convex_hull: image (int n*m matrix). The convex hull of the image received in input
    """

    # Finding contours for the image
    im, contour_list, hierarchy = cv2.findContours(image=image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # create hull array for convex hull points
    hull_list = []

    # calculate points for each contour
    for contour in contour_list:
        # creating convex hull object for each contour
        hull_list.append(cv2.convexHull(contour, False))

    # create an empty black image
    return_image = np.zeros((image.shape[0], image.shape[1]), np.uint8)

    # draw contours and hull points in the empty image
    for i in range(len(contour_list)):
        color = 255
        # draw ith convex hull object
        cv2.drawContours(return_image, hull_list, i, color, -1, 8)

    return return_image


def find_peaks(image, threshold=3):
    """find and return intensity peaks in the given image.

    Parameters:
    image: image (int n*m matrix). Image of the veins
    threshold: positive int. The threshold value for determing if a pixel value is a peak. default is 3

    Returns:
    peak_list: list. The intensity values where peaks occur (decreasing order)
    """
    # distribution of pixel intensity [0-255] in image, peak values
    hist, bins = np.histogram(image.ravel(), 256, [0, 256])

    indexes = scipy.signal.argrelextrema(
        np.array(hist),
        comparator=np.greater, order=threshold
    )

    # reverse order to decreasing
    peak_list = indexes[0][::-1]
    peak_list = np.append(peak_list, 0)
    peak_list = peak_list.tolist()
    return peak_list


def generate_target_candidates(image, mask):
    """ Given a cropped veins image, it generate a set of
       candidate extensions of the mask (more pixels in white).

       Parameters:
       image: image (int n*m matrix). grayscale image of (the center of) the forearm
       mask: image (int n*m matrix). grayscale mask of the veins

       Returns:
       target_mask_list: python list of grayscale images (int n*m matrix). The set of possible tracking masks
       """

    # generate the convex hull of the mask (include pixels in concave parts)
    mask_hull = convex_hull(mask)

    # get the list of intensity peaks in the image
    peak_list = find_peaks(image)

    # list holding results
    intensity_masks_list = []

    # for each intensity peak value generate an intensity mask
    for peak in peak_list:
        # from the mask select only pixels higher than the peak and set those to 255
        _, binary = cv2.threshold(image, peak, 255, cv2.THRESH_BINARY)

        # generate the convex hull (include pixels in concave parts)
        threshold_hull = convex_hull(binary)

        # fuse the two convex hulls
        and_mask = np.bitwise_or(mask_hull, threshold_hull)

        # select only pixels from image that lies in the and_mask area
        intensity_mask = np.bitwise_and(image, and_mask)

        # add the intensity mask to the return list
        intensity_masks_list.append(intensity_mask)

    return intensity_masks_list


def first_align(orig, mask, crop):
    """ Search for the target tracking image and find it's displacement in the original image

        Parameters:
        orig: image. Image obtained from the acquisition system
        mask:  image. deformed squared mask of the veins
        crop:  image. Deformed squared version of "orig" with same dims as "mask"

        Returns:
        mask_def: image. Vein mask with same aspect ration of "orig"
        tracking: image. mask to be used in subsequent tracking
        ret_x: positive int. position on X axis of "tracking" inside "orig"
        ret_y: positive int. position on Y axis of "tracking" inside "orig"
    """

    # sanity check that the params exists and eventually transform in grayscale
    assert orig is not None
    assert mask is not None
    assert crop is not None

    if (len(orig.shape) == 3 and orig.shape[2] > 1):
        image_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = orig
    if (len(crop.shape) == 3 and crop.shape[2] > 1):
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        crop_gray = crop
    if (len(mask.shape) == 3 and mask.shape[2] > 1):
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask


    # find the ROI from the mask and store its coordinates
    roi_vals = np.array(np.nonzero(mask_gray))
    roi_coords = [roi_vals[0].min(), roi_vals[0].max(), roi_vals[1].min(), roi_vals[1].max()]

    # crop the ROI from the mask and from the square image
    mask_roi = mask_gray[roi_coords[0]:roi_coords[1], roi_coords[2]:roi_coords[3]]
    crop_roi = crop_gray[roi_coords[0]:roi_coords[1], roi_coords[2]:roi_coords[3]]

    # generate the target candidates
    target_list = generate_target_candidates(crop_roi, mask_roi)

    # store the coordinates of the target patch inside the square image
    ret_x, ret_y = 0, 0
    tracking_target = None


    # evaluate each candidate target from smaller to bigger, return the first that properly align
    for target in target_list:
        # search for target position in square image and save
        result = match_template(crop_gray, target)
        ij = np.unravel_index(np.argmax(result), result.shape)
        ret_x, ret_y = ij[::-1]

        # check the overall error; roi_coords[2] and roi_coords[0] are the target X & Y true displacement
        delta_x = abs(ret_x - roi_coords[2])
        delta_y = abs(ret_y - roi_coords[0])

        # if the error is under 2 pixels select the target and stop the loop
        if delta_x < 2 and delta_y < 2:
            tracking_target = target
            break


    # deform the target tracking and the mask to match initial input image form factor
    new_x = int(tracking_target.shape[1] * (image_gray.shape[0] / crop_gray.shape[0]))
    new_y = int(tracking_target.shape[0] * (image_gray.shape[0] / crop_gray.shape[0]))

    tracking = cv2.resize(tracking_target, dsize=(new_x, new_y), interpolation=cv2.INTER_CUBIC)
    mask_def = cv2.resize(mask_roi, dsize=(new_x, new_y), interpolation=cv2.INTER_CUBIC)


    # find the coordinates in the initial full_res image
    match = match_template(image_gray, tracking)
    ij = np.unravel_index(np.argmax(match), match.shape)
    ret_x, ret_y = ij[::-1]

    # return the (deformed) initial mask, the tracking mask and it's coordinates in the initial image
    return mask_def, tracking, ret_x, ret_y
