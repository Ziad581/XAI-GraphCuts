import torch
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import shap
import maxflow

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model_path = "/Users/ziad/Desktop/untitled folder/GoogLeNetAugmented.pt"
model = torch.load(model_path, map_location=device)
model.to(device)
model.eval()

# Paths
img_path = '/Users/ziad/Desktop/test2/5.jpg'

saliency_save_path = img_path.replace('.jpg', '_S_.jpg')

binary_mask_saliency_path = img_path.replace('.jpg', '_S_mask.jpg')

gradcam_save_path = img_path.replace('.jpg', '_G_.jpg')

binary_gradcam_save_path = img_path.replace('.jpg', '_G_mask.jpg')

mask_save_path = img_path.replace('.jpg', '_lime_mask.jpg')

binary_mask_shap_path = img_path.replace('.jpg', '_Shap_mask.jpg')


# Preprocess the image
def preprocess(image, size=224):
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(image)


def transform (image, size=224):
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image)


# Load and preprocess the image
img = Image.open(img_path)
X = preprocess(img)


def load_image(image_paths):
    image = Image.open(image_paths).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)


test_image = load_image(img_path)
test_image_np = test_image.cpu().numpy()[0]  # Remove batch dimension
test_image_np = np.transpose(test_image_np, (1, 2, 0))  # Move channel to the end
test_image_np = (test_image_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])  # Denormalize
test_image_np = np.clip(test_image_np, 0, 1)  # Clip to the valid range of image values

# Get predicted classes
with torch.no_grad():
    outputs = model(X.to(device))
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    top_prob, top_catid = torch.topk(probabilities, 4)
    top_classes = top_catid[0].cpu().numpy()
    top_probabilities = top_prob[0].cpu().numpy()

print("Top 4 predicted classes:")
for i, (cls, prob) in enumerate(zip(top_classes, top_probabilities)):
    print(f"{i + 1}: Class ID {cls+1} with probability {prob}")

try:
    class_choice = int(input("Select a class to explain (1-4): "))
    assert 1 <= class_choice <= 4
    selected_class = top_classes[class_choice - 1]
except (ValueError, AssertionError):
    print("Invalid input. Defaulting to the most probable class.")
    selected_class = top_classes[0]


# Saliency Map Generation
def generate_saliency_map(model, X, selected_class):
    X.requires_grad_()
    outputs = model(X)
    model.zero_grad()
    outputs[0, selected_class].backward()
    saliency, _ = torch.max(X.grad.data.abs(), dim=1)
    saliency = saliency.squeeze().cpu().numpy()
    saliency = saliency / saliency.max()
    saliency = Image.fromarray(np.uint8(255 * saliency))
    saliency = saliency.resize(img.size, Image.LANCZOS)
    return saliency


# Generate and save saliency map
saliency = generate_saliency_map(model, X, selected_class)
saliency.save(saliency_save_path)

# Load the saliency map
saliency_map = cv2.imread(saliency_save_path, cv2.IMREAD_GRAYSCALE)


# Apply Otsu's thresholding and morphological operations
def apply_threshold_and_morphology(image, kernel_size=1):
    _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    return binary_mask


# Generate binary mask from saliency map
binary_mask_saliency = apply_threshold_and_morphology(saliency_map)
cv2.imwrite(binary_mask_saliency_path, binary_mask_saliency)


# Grad-CAM Generation
def generate_grad_cam(model, image, target_layer_name, selected_class):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = transform(image).unsqueeze(0).to(device)
    target_layer = getattr(model, target_layer_name)
    activations = None
    gradients = None

    def hook_fn(module, input, output):
        nonlocal activations
        activations = output

    def grad_hook_fn(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    hook = target_layer.register_forward_hook(hook_fn)
    hook_grad = target_layer.register_full_backward_hook(grad_hook_fn)
    outputs = model(input_image)
    model.zero_grad()
    outputs[0, selected_class].backward()
    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    activation_map = torch.sum(weights * activations, dim=1, keepdim=True)
    activation_map = torch.nn.functional.relu(activation_map)
    heatmap = activation_map.squeeze().detach().cpu().numpy()
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (image.width, image.height))
    heatmap = np.uint8(255 * heatmap)
    _, binary_mask_gradcam = cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hook.remove()
    hook_grad.remove()
    return heatmap, binary_mask_gradcam


# Generate and save Grad-CAM
heatmap, binary_mask_gradcam = generate_grad_cam(model, img, "inception5b", selected_class)
cv2.imwrite(gradcam_save_path, heatmap)
cv2.imwrite(gradcam_save_path.replace('.jpg', '_mask.jpg'), binary_mask_gradcam)


# LIME Explanation Generation
def generate_lime_explanation(model, image, transform, selected_class, device):
    def predict_func(images):
        images = [transform(Image.fromarray(np.uint8(img)).convert('RGB')).unsqueeze(0).to(device) for img in images]
        batch = torch.cat(images, dim=0)
        with torch.no_grad():
            outputs = model(batch)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
        return probabilities

    test_image_np = np.array(image)
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(test_image_np, predict_func, top_labels=4, hide_color=0, num_samples=1500,
                                             segmentation_fn=SegmentationAlgorithm('quickshift', kernel_size=9, max_dist=200, ratio=0.7))
    temp, mask = explanation.get_image_and_mask(selected_class, positive_only=True, num_features=2, hide_rest=True)

    highlighted = test_image_np.copy()
    highlighted[mask == 1] = highlighted[mask == 1] * 0.5 + np.array([0, 255, 0],
                                                                     dtype=np.uint8) * 0.5  # Overlay green color
    # highlighted[mask == 0] = highlighted[mask == 0] * 0.5 + np.array([255, 0, 0], dtype=np.uint8) * 0.5  # Overlay red   color
    fig, (ax2) = plt.subplots(1, 1, figsize=(3, 3))
    ax2.imshow(highlighted, interpolation='nearest')
    ax2.set_title(f'Explanation for Class: {selected_class + 1}')

    segments = explanation.segments
    binary_mask_lime = np.zeros(segments.shape, dtype=np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                binary_mask_lime[segments == segments[i, j]] = 255
    return binary_mask_lime


# Generate and save LIME mask
binary_mask_lime = generate_lime_explanation(model, img, transform, selected_class, device)
cv2.imwrite(mask_save_path, binary_mask_lime)


# SHAP Explanation Generation
def generate_shap_explanation(model, image_path, background_data_path, selected_class, device):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def load_image(image_path):
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        return image

    image = load_image(image_path)
    dataset = datasets.ImageFolder(root=background_data_path, transform=transform)
    background_data, _ = next(iter(DataLoader(dataset, batch_size=50, shuffle=True)))
    background_data = background_data.to(device)

    explainer = shap.GradientExplainer(model, background_data)
    shap_values = explainer.shap_values(image, nsamples=100)
    combined_shap_values = np.sum(np.abs(shap_values), axis=0)
    threshold = np.percentile(combined_shap_values, 60)
    thresholded_shap_values = np.where(combined_shap_values > threshold, combined_shap_values, 0)
    binary_mask_shap = np.where(thresholded_shap_values[0] > 0, 255, 0).astype(np.uint8)[0]
    kernel = np.ones((1, 1), np.uint8)
    binary_mask_shap = cv2.morphologyEx(binary_mask_shap, cv2.MORPH_CLOSE, kernel)
    return binary_mask_shap


# Generate and save SHAP mask
background_data_path = '/Users/ziad/Downloads/BA/DatasetA/test'
binary_mask_shap = generate_shap_explanation(model, img_path, background_data_path, selected_class, device)
cv2.imwrite(binary_mask_shap_path, binary_mask_shap)


# Combine masks using SHAP as baseline and expand with others
def combine_masks_iterative_refinement(base_mask, other_masks, iterations=10):
    combined_mask = base_mask.copy()
    kernel = np.ones((1, 1), np.uint8)  # Kernel size for dilation
    for _ in range(iterations):
        for mask in other_masks:
            dilated_mask = cv2.dilate(mask, kernel, iterations=10)
            combined_mask = np.maximum(combined_mask, dilated_mask)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel= np.ones((3, 3), np.uint8))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    return combined_mask


# Graph Cut Segmentation
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
            g.add_tedge(nodeids[i, j], source_weight, sink_weight)

    # Set up edge weights
    structure = np.array([[2, 2, 2], [2, 0, 2], [2, 2, 2]])
    g.add_grid_edges(nodeids, structure=structure, symmetric=True)

    # Compute the max-flow/min-cut and get the segments
    g.maxflow()

    sgm = g.get_grid_segments(nodeids)  # True for source set, False for sink set
    return np.int_(sgm).reshape(image.shape[:2])


# Apply segmentation mask to color image
def apply_segmentation_mask_to_color_image(image, segmentation_mask):
    """Apply the segmentation mask to the original color image to separate foreground and background."""
    mask = np.dstack([segmentation_mask] * 3)  # Convert mask to 3 channels
    foreground_image = np.where(mask == 0, image, 255)
    background_image = np.where(mask == 1, image, 255)
    return foreground_image, background_image


# Combine masks and segment the image
masks = [binary_mask_gradcam,binary_mask_lime]
#masks = [binary_mask_gradcam, binary_mask_shap]
combined_mask = combine_masks_iterative_refinement(binary_mask_shap, masks)
segmented_mask = segment_image_with_graph_cut(test_image_np, combined_mask)
foreground_image, background_image = apply_segmentation_mask_to_color_image(test_image_np, segmented_mask)


# Display the results
fig, axs = plt.subplots(2, 5, figsize=(10, 5))

# Row 2
axs[0, 0].imshow(binary_mask_saliency, cmap='gray')
axs[0, 0].set_title("Saliency Binary Mask")

axs[0, 1].imshow(binary_mask_gradcam, cmap='gray')
axs[0, 1].set_title("Grad-CAM Binary Mask")

axs[0, 2].imshow(binary_mask_lime, cmap='gray')
axs[0, 2].set_title("LIME Binary Mask")

axs[0, 3].imshow(binary_mask_shap, cmap='gray')
axs[0, 3].set_title("SHAP Binary Mask")

axs[0, 4].imshow(combined_mask, cmap='gray')
axs[0, 4].set_title("Combined Mask")

# Row 3
axs[1, 0].imshow(test_image_np)
axs[1, 0].set_title("Original Image")

axs[1, 1].imshow(foreground_image)
axs[1, 1].set_title("Foreground")

axs[1, 2].imshow(background_image)
axs[1, 2].set_title("Background")

for ax in axs.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()
