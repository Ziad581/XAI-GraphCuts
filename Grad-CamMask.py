import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

def generate_grad_cam_and_mask(model_path, image_path, target_layer_name, output_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path)
    input_image = transform(image).unsqueeze(0)

    target_layer = getattr(model, target_layer_name)

    activations = None
    gradients = None

    def hook_fn(module, input, output):
        nonlocal activations
        activations = output

    hook = target_layer.register_forward_hook(hook_fn)

    def grad_hook_fn(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    hook_grad = target_layer.register_full_backward_hook(grad_hook_fn)  # Updated here

    output = model(input_image)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    top_prob, top_catid = torch.topk(probabilities, 4)
    print("Top 4 predicted classes:")
    for i in range(top_prob.size(1)):
        print(f"{i + 1}: Class ID {top_catid[0][i].item()} with probability {top_prob[0][i].item()}")

    try:
        class_choice = int(input("Select a class to explain (1-4): "))
        assert 1 <= class_choice <= 4
        selected_class = top_catid[0][class_choice - 1].item()
    except (ValueError, AssertionError):
        print("Invalid input. Defaulting to the most probable class.")
        selected_class = top_catid[0][0].item()

    model.zero_grad()
    output[0, selected_class].backward()

    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    activation_map = torch.sum(weights * activations, dim=1, keepdim=True)

    activation_map = nn.functional.relu(activation_map)
    heatmap = activation_map.squeeze().detach().numpy()
    heatmap /= np.max(heatmap)  # Normalize the heatmap

    # Resize heatmap to the original image size
    heatmap = cv2.resize(heatmap, (image.width, image.height))
    # Convert heatmap to 8-bit format
    heatmap = np.uint8(255 * heatmap)
    # Apply Otsu's thresholding to create a binary mask
    _, binary_mask = cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Save the binary mask
    binary_mask_path = output_path.replace('.jpg', '_mask.jpg')
    cv2.imwrite(binary_mask_path, binary_mask)

    #superimposed image using the original heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(np.array(image), 0.7, heatmap, 0.3, 0)
    cv2.imwrite(output_path, superimposed_img)

    hook.remove()
    hook_grad.remove()


model_path = "/Users/ziad/Desktop/untitled folder/GoogLeNetAugmented.pt"
image_path = '/Users/ziad/Downloads/LatexCurrent/img/XAI/3/3.jpg'
target_layer_name = "inception5b"
output_path = '/Users/ziad/Downloads/LatexCurrent/img/XAI/3/3g.jpg'

generate_grad_cam_and_mask(model_path, image_path, target_layer_name, output_path)








