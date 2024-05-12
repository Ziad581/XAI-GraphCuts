import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image

model = torch.load("/Users/ziad/Desktop/untitled folder/GoogLeNetAugmented.pt")


img = Image.open('/Users/ziad/Downloads/LatexCurrent/img/XAI/3/3.jpg')
save_path = "/Users/ziad/Desktop/3s.jpg"


# Preprocess the image
def preprocess(image, size=224):
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(image)


# Preprocess the image
X = preprocess(img)


model.eval()

X.requires_grad_()

# forward pass through model to get the scores
scores = model(X)
# Get the index corresponding to the maximum score
#score_max_index = 2
score_max_index = scores.argmax()
# backward function on scores[score_max_index] performs the backward pass
scores[0, score_max_index].backward()

#gradient with respect to the input image 
saliency, _ = torch.max(X.grad.data.abs(), dim=1)

# Resize the saliency map to match the input image dimensions
saliency = saliency.squeeze().cpu().numpy()  # Convert to a 2D array
saliency = saliency / saliency.max()  # Normalize to [0, 1]
saliency = Image.fromarray(np.uint8(255 * saliency))

# Resize the saliency map to match the input image dimensions
saliency = saliency.resize(img.size, Image.LANCZOS)

# Save the saliency map as an image without displaying it
saliency.save(save_path)


import cv2
import numpy as np

# Load the saliency map
saliency_map = cv2.imread(save_path, cv2.IMREAD_GRAYSCALE)

# Apply Otsu's thresholding to automatically find the threshold value
_, binary_mask = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# Define a function to apply morphological operations with a variable kernel size
def apply_morphological_operations(binary_mask, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    return binary_mask


# Experiment with different kernel sizes
kernel_size = 1  # Example, adjust based on experimentation
binary_mask = apply_morphological_operations(binary_mask, kernel_size)

# Save 
cv2.imwrite("/Users/ziad/Desktop/3sb.png", binary_mask)