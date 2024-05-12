import torch
from torchvision import transforms, datasets
from PIL import Image
import shap
from torch.utils.data import DataLoader
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_image(image_path):
    """Loads an image and applies transformations."""
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)



model = torch.load("/Users/ziad/Desktop/untitled folder/GoogLeNetAugmented.pt").to(device)
model.eval()

# Load the image for analysis
image_path = '/Users/ziad/Downloads/LatexCurrent/img/XAI/3/3.jpg'
test_image = load_image(image_path)

# Background data for SHAP
data_dir = '/Users/ziad/Desktop/BA/DatasetA/test'
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
background_data, _ = next(iter(DataLoader(dataset, batch_size=50, shuffle=True)))
background_data = background_data.to(device)

# SHAP explanation
explainer = shap.GradientExplainer(model, background_data)
shap_values = explainer.shap_values(test_image, nsamples=100)

# Combine and threshold SHAP values
combined_shap_values = np.sum(np.abs(shap_values), axis=0)
threshold = np.percentile(combined_shap_values, 60)
thresholded_shap_values = np.where(combined_shap_values > threshold, combined_shap_values, 0)

# Convert to binary mask
binary_mask = np.where(thresholded_shap_values[0] > 0, 255, 0).astype(np.uint8)[0]

# Refine binary mask with morphological operations
kernel = np.ones((1, 1), np.uint8)
#binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)


# Save the binary mask for segmentation
cv2.imwrite('/Users/ziad/Downloads/LatexCurrent/img/XAI/3/3shapb.jpg', binary_mask)
