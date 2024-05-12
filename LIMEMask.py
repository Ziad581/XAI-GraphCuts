import torch
from torchvision import transforms
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Model loading
model_path = '/Users/ziad/Desktop/untitled folder/GoogLeNetAugmented.pt'
model = torch.load(model_path, map_location=device)
model.to(device)
model.eval()

# Prediction function adjustment for LIME
def predict_func(images):
    images = [transform(Image.fromarray(np.uint8(img)).convert('RGB')).unsqueeze(0).to(device) for img in images]
    batch = torch.cat(images, dim=0)
    with torch.no_grad():
        outputs = model(batch)
    probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
    return probabilities

# Load an image and prepare for explanation
test_image_path = '/Users/ziad/Downloads/LatexCurrent/img/XAI/3/3.jpg'
test_image = Image.open(test_image_path).convert('RGB')
test_image_np = np.array(test_image)

# Explanation setup
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(test_image_np, predict_func, top_labels=4, hide_color=0, num_samples=2000,
                                         segmentation_fn=SegmentationAlgorithm('quickshift', kernel_size=9, max_dist=200, ratio=0.7))

top_classes = explanation.top_labels
print("Top predicted classes:")
for idx, cls in enumerate(top_classes):
    print(f"{idx + 1}: Stage {cls + 1}")

# User selects a class to explain
try:
    class_choice = int(input("Select a class to explain (1-4): "))
    assert 1 <= class_choice <= 4
    selected_class = top_classes[class_choice - 1]
except (ValueError, AssertionError):
    print("Invalid input. Defaulting to the top predicted class.")
    selected_class = top_classes[0]

# Visualization of the explanation for the specified class
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
ax1.imshow(test_image)
ax1.set_title('Original Image')

temp, mask = explanation.get_image_and_mask(selected_class, positive_only=True, num_features=2, hide_rest=True)

# Overlay mask manually
highlighted = test_image_np.copy()
highlighted[mask == 1] = highlighted[mask == 1] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5  # Overlay green color
#highlighted[mask == 0] = highlighted[mask == 0] * 0.5 + np.array([255, 0, 0], dtype=np.uint8) * 0.5  # Overlay red   color

ax2.imshow(highlighted, interpolation='nearest')
ax2.set_title(f'Explanation for Class: {selected_class + 1}')

segments = explanation.segments
ax3.imshow(segments, cmap='viridis')
ax3.set_title('Superpixel Segments')

plt.show()

binary_mask = np.zeros(segments.shape, dtype=np.uint8)
for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if mask[i, j]:
            binary_mask[segments == segments[i, j]] = 255  # Mark as foreground

#Morphological operations to refine the mask
kernel = np.ones((3, 3), np.uint8)
#binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
#binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

# Save the binary mask
mask_save_path = test_image_path.replace('.jpg', '_lime_mask.jpg')
cv2.imwrite(mask_save_path, binary_mask)

print(f"LIME mask saved to: {mask_save_path}")

