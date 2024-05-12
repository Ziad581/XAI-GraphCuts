import numpy as np
import cv2
import matplotlib.pyplot as plt
import maxflow
from skimage.filters import threshold_otsu

def segment_image_with_graph_cut(image, mask):
    """Segment the image using the Graph Cut algorithm, utilizing the mask directly."""
    rows, cols = image.shape[:2]
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((rows, cols))

    # Normalize the mask to range [0, 1] and use it directly to set up terminal weights
    normalized_mask = mask / 255.0  # Assuming mask is in 0-255 range
    for i in range(rows):
        for j in range(cols):
            source_weight = normalized_mask[i, j]
            sink_weight = (1 - source_weight)
            if (i == 61 and j == 127):
                print(f"Pixel at ({i},{j}) - Source weight: {source_weight}, Sink weight: {sink_weight}")
            g.add_tedge(nodeids[i, j], source_weight, sink_weight)

    # Set up edge weights
    structure = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    g.add_grid_edges(nodeids, structure=structure, symmetric=True)

    # Compute the max-flow/min-cut and get the segments
    g.maxflow()

    sgm = g.get_grid_segments(nodeids)  # True for source set, False for sink set
    return np.int_(sgm).reshape(image.shape[:2])


def combine_masks(masks):
    """Combine multiple masks using Otsu's thresholding."""
    masks = [mask.astype(np.uint8) for mask in masks]
    stacked_masks = np.stack(masks, axis=0)
    summed_masks = np.sum(stacked_masks, axis=0).astype(np.int64)
    # Apply Otsu's thresholding to find the optimal threshold to separate foreground and background
    if summed_masks.max() > 0:
        otsu_threshold = threshold_otsu(summed_masks)
    else:
        otsu_threshold = 0
    combined_mask = summed_masks > otsu_threshold
    return combined_mask.astype(np.uint8) * 255


def apply_segmentation_mask_to_color_image(image, segmentation_mask):
    """Apply the segmentation mask to the original color image to separate foreground and background."""
    mask = np.dstack([segmentation_mask] * 3)  # Convert mask to 3 channels
    foreground_image = np.where(mask == 0, image, 255)
    background_image = np.where(mask == 1, image, 255)
    return foreground_image, background_image


def load_and_segment(image_path, mask_paths):
    """Load an image and multiple masks, then perform segmentation using a weighted combination of masks."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load the main image from {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = []
    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to load mask from {mask_path}")
        masks.append(mask)

    combined_mask = combine_masks(masks)
    print("Combined mask values range:", combined_mask.min(), combined_mask.max())

    cleaned_segmented_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    cleaned_segmented_mask = cv2.morphologyEx(cleaned_segmented_mask.astype(np.uint8), cv2.MORPH_CLOSE,np.ones((5, 5), np.uint8))

    segmented_mask = segment_image_with_graph_cut(image, combined_mask)
    foreground_image, background_image = apply_segmentation_mask_to_color_image(image, segmented_mask)

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(foreground_image)
    plt.title('Foreground')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(background_image)
    plt.title('Background')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(combined_mask, cmap='gray')
    plt.title('Combined Mask')
    plt.axis('off')
    plt.show()



if __name__ == "__main__":
    image_path = '/Users/ziad/Downloads/LatexCurrent/img/XAI/5/5.jpg'  # Update this path
    mask_paths = [
        '/Users/ziad/Downloads/LatexCurrent/img/XAI/5/5shapb.jpg',
        '/Users/ziad/Downloads/LatexCurrent/img/XAI/5/5g_mask.jpg',
        '/Users/ziad/Downloads/LatexCurrent/img/XAI/5/5_lime_mask11.jpg',
    ]


    load_and_segment(image_path, mask_paths)



