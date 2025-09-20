import numpy as np
import cv2
from PIL import Image
from skimage import filters, exposure
from skimage.segmentation import quickshift
from skimage.measure import label, find_contours
from scipy.ndimage import mean

def refine_segmentation(image):
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return closing

def darken_image2(image, darken_factor=0.8):
    image_np = np.array(image)
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    max_intensity = 255
    transform_curve = np.array(
        [(i / max_intensity) ** (1 / darken_factor) * max_intensity for i in range(256)],
        dtype=np.uint8
    )
    darkened_image = cv2.LUT(image_np, transform_curve)
    return darkened_image

def merge_regions(labeled_image, original_image, intensity_threshold):
    unique_labels = np.unique(labeled_image)
    regions_mean_intensity = mean(original_image, labels=labeled_image, index=unique_labels)
    label_to_index = {label: index for index, label in enumerate(unique_labels)}

    merged = False
    for region_label in unique_labels:
        if region_label == 0:
            continue
        region_mean = regions_mean_intensity[label_to_index[region_label]]
        region_mask = labeled_image == region_label
        contours = find_contours(region_mask, 0.01)
        for contour in contours:
            for coord in contour.astype(int):
                neighbor_label = labeled_image[coord[0], coord[1]]
                if neighbor_label != region_label and neighbor_label != 0:
                    neighbor_index = label_to_index[neighbor_label]
                    if abs(regions_mean_intensity[neighbor_index] - region_mean) < intensity_threshold:
                        labeled_image[region_mask] = neighbor_label
                        regions_mean_intensity[neighbor_index] = mean(
                            original_image, labels=labeled_image, index=[neighbor_label]
                        )[0]
                        merged = True
                        break
            if merged:
                break
    return labeled_image if merged else None

def ignore_black_regions(image, threshold=0.05):
    mask = image > threshold
    return mask

def segmentation_and_merge(img, sigma=1, kernel_size=3, max_dist=8, ratio=0.5,
                           clip_limit=0.5, intensity_threshold=0.05):
    mask = ignore_black_regions(img)

    img_smoothed = filters.gaussian(img, sigma=sigma)
    img_edges = filters.sobel(img_smoothed)
    img_preprocessed = img_smoothed + img_edges
    img_preprocessed = (img_preprocessed - np.min(img_preprocessed)) / (
        np.max(img_preprocessed) - np.min(img_preprocessed)
    )
    img_eq = exposure.equalize_adapthist(img_preprocessed, clip_limit=clip_limit)

    segments_quick = quickshift(img_eq, kernel_size=kernel_size,
                                max_dist=max_dist, ratio=ratio, convert2lab=False)

    labeled_segments = label(segments_quick * mask)

    while True:
        new_labeled_segments = merge_regions(
            labeled_segments, img * mask, intensity_threshold=intensity_threshold
        )
        if new_labeled_segments is None:
            break
        labeled_segments = new_labeled_segments

    return labeled_segments
