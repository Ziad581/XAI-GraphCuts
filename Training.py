import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import googlenet, GoogLeNet_Weights
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import ConcatDataset
from sklearn.metrics import f1_score
import multiprocessing
import time
import warnings

warnings.filterwarnings("ignore", message="Named tensors and all their associated APIs are an experimental feature")

start_time = time.time()

classes = ('class1', 'class2', 'class3', 'class4')

if __name__ == '__main__':
    multiprocessing.freeze_support()

torchvision_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

train_data_path = '/Users/ziad/Desktop/BA/DatasetA/PiiD'
test_data_path = '/Users/ziad/Desktop/test_augmented'

train_dataset = torchvision.datasets.ImageFolder(root=train_data_path, transform=torchvision_transforms)
test_dataset = torchvision.datasets.ImageFolder(root=test_data_path, transform=torchvision_transforms)
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

net = googlenet(weights=GoogLeNet_Weights.DEFAULT)
# Modify the final fully connected layer
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, len(classes))

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.90, weight_decay=0.01)


patience = 6  # number of epochs to wait before stopping
best_loss = float('inf')
tolerance_threshold = 0.005
epochs_no_improve = 0

try:
    for epoch in range(27):
        running_loss = 0.0
        net.train()

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Average Training Loss: {average_loss}')

        # Validation step
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_data in test_loader:
                val_inputs, val_labels = val_data
                val_outputs = net(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()

        # average loss
        average_val_loss = val_loss / len(test_loader)
        print(f'Epoch {epoch + 1}, Average Validation Loss: {average_val_loss}')

        # Check for early stopping
        if average_loss < best_loss - tolerance_threshold:
            best_loss = average_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping triggered after epoch {epoch + 1}')
                break

except KeyboardInterrupt:
    print("Training interrupted by user.")

torch.save(net, '/Users/ziad/Desktop/GoogLeNetAug.pt')

# Testing
class_correct = list(0. for _ in range(len(classes)))
class_total = list(0. for _ in range(len(classes)))
class_predictions = [[] for _ in range(len(classes))]
class_labels = [[] for _ in range(len(classes))]

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        correct_predictions = (predicted == labels)

        if not isinstance(correct_predictions, torch.Tensor):
            raise TypeError("Expected 'correct_predictions' to be a PyTorch tensor")

        for i in range(labels.size(0)):
            label = labels[i].item()
            is_correct = correct_predictions[i].item()
            class_correct[label] += is_correct
            class_total[label] += 1
            class_predictions[label].append(predicted[i].item())
            class_labels[label].append(label)

# Accuracy and F1 score for each class
f1_scores = []
for i in range(len(classes)):
    accuracy = 100 * class_correct[i] / class_total[i]
    f1 = f1_score(class_labels[i], class_predictions[i], average='weighted')
    print(f'Accuracy of {classes[i]}: {accuracy:.2f}%, F1 Score: {f1:.4f}')
    f1_scores.append(f1)

# overall accuracy and f1
correct = sum(class_correct)
total = sum(class_total)
overall_accuracy = 100 * correct / total
overall_f1_score = np.mean(f1_scores)
print(f"Overall accuracy: {overall_accuracy:.2f}%, Overall F1 Score: {overall_f1_score:.4f}")


end_time = time.time()
elapsed_time = end_time - start_time
elapsed_minutes = elapsed_time / 60
print(f"Training finished in {elapsed_minutes:.2f} minutes.")
