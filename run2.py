import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset

# Directories for training and validation data
train_dir = 'Training'
validation_dir = 'Validation'

# Image dimensions
img_size = 64

# Load images and labels
def load_images_and_labels(directory, max_images=None):
    images = []
    labels = []
    for label in ['male', 'female']:
        path = os.path.join(directory, label)
        class_num = 0 if label == 'male' else 1
        img_names = os.listdir(path)
        if max_images:
            img_names = img_names[:max_images]
        for img_name in tqdm(img_names):
            try:
                img_path = os.path.join(path, img_name)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (img_size, img_size))
                images.append(image)
                labels.append(class_num)
            except Exception as e:
                print(f"Failed to process image {img_path}: {e}")
    print(f"Loaded {len(images)} images from {directory}")
    return np.array(images), np.array(labels)

# Load training and validation data (use a smaller subset for testing)
train_images, train_labels = load_images_and_labels(train_dir, max_images=1000)
test_images, test_labels = load_images_and_labels(validation_dir, max_images=1000)

# Convert RGB to grayscale
train_images_gray = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in train_images])
test_images_gray = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in test_images])

# Normalize images
train_images_gray = train_images_gray / 255.0
test_images_gray = test_images_gray / 255.0

# Flatten images for SVM
train_images_gray_flat = train_images_gray.reshape(len(train_images_gray), -1)
test_images_gray_flat = test_images_gray.reshape(len(test_images_gray), -1)

# Split training data into training and validation
X_train, X_val, y_train, y_val = train_test_split(train_images_gray_flat, train_labels, test_size=0.2, random_state=42)

# First Experiment: Train an SVM model
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Test the SVM model
y_pred = svm.predict(test_images_gray_flat)
conf_matrix_svm = confusion_matrix(test_labels, y_pred)
f1_svm = f1_score(test_labels, y_pred, average='weighted')

# Print SVM results
print("SVM Confusion Matrix:")
print(conf_matrix_svm)
print("\nSVM Classification Report:")
print(classification_report(test_labels, y_pred))
print(f"SVM Average F1 Score: {f1_svm}")

print("------------------------------------Neural Network----------------------------------------")
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(img_size * img_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(img_size * img_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

# Instantiate the models
model_1 = Model1()
model_2 = Model2()

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer_1 = optim.Adam(model_1.parameters())
optimizer_2 = optim.Adam(model_2.parameters())

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
test_images_gray_tensor = torch.tensor(test_images_gray_flat, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32)

# Define datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataset = TensorDataset(test_images_gray_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training loop

def train_model(model, optimizer, criterion, train_loader, val_loader, num_epochs=10):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

            # Calculate accuracy
            predicted = outputs.round().squeeze()
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item() * inputs.size(0)

                # Calculate accuracy
                predicted = outputs.round().squeeze()
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    return train_losses, val_losses, train_accuracies, val_accuracies


# Train the models
train_losses_1, val_losses_1, train_accuracies_1, val_accuracies_1 = train_model(model_1, optimizer_1, criterion,
                                                                                 train_loader, val_loader)
train_losses_2, val_losses_2, train_accuracies_2, val_accuracies_2 = train_model(model_2, optimizer_2, criterion,
                                                                                 train_loader, val_loader)

# Plot the training and validation losses and accuracies
plt.figure(figsize=(20, 10))

# Plot losses
plt.subplot(2, 1, 1)
plt.plot(train_losses_1, label='Model 1 Train Loss')
plt.plot(val_losses_1, label='Model 1 Validation Loss')
plt.plot(train_losses_2, label='Model 2 Train Loss')
plt.plot(val_losses_2, label='Model 2 Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot accuracies
plt.subplot(2, 1, 2)
plt.plot(train_accuracies_1, label='Model 1 Train Accuracy')
plt.plot(val_accuracies_1, label='Model 1 Validation Accuracy')
plt.plot(train_accuracies_2, label='Model 2 Train Accuracy')
plt.plot(val_accuracies_2, label='Model 2 Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
# Save the best model
torch.save(model_1.state_dict(), 'best_model_1.pth')

# Reload the best model
best_model = Model1()
best_model.load_state_dict(torch.load('best_model_1.pth'))
best_model.eval()

# Test the best model
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()
    return correct / total

test_accuracy_1 = test_model(model_1, val_loader)
test_accuracy_2 = test_model(model_2, val_loader)

print("Model 1 Test Accuracy:", test_accuracy_1)
print("Model 2 Test Accuracy:", test_accuracy_2)

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()
            all_preds.append(predicted)
            all_labels.append(labels.unsqueeze(1))
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return correct / total, confusion_matrix(all_labels, all_preds), f1_score(all_labels, all_preds, average='weighted')

test_accuracy, conf_matrix_nn, f1_nn = test_model(best_model, test_loader)

print("Neural Network Confusion Matrix:")
print(conf_matrix_nn)
#print("\nNeural Network Classification Report:")
#print(classification_report(test_labels, torch.cat([torch.round(best_model(torch.tensor(test_images_gray_tensor[i:i+1]))).numpy() for i in range(len(test_images_gray_tensor))])))

print(f"Neural Network Average F1 Score: {f1_nn}")

print("--------------------------------------------Convolutional Neural Networks----------------------------------")


# Load images and labels
def load_images_and_labels(directory, max_images=None):
    images = []
    labels = []
    for label in ['male', 'female']:
        path = os.path.join(directory, label)
        class_num = 0 if label == 'male' else 1
        img_names = os.listdir(path)
        if max_images:
            img_names = img_names[:max_images]
        for img_name in tqdm(img_names):
            try:
                img_path = os.path.join(path, img_name)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (img_size, img_size))
                images.append(image)
                labels.append(class_num)
            except Exception as e:
                print(f"Failed to process image {img_path}: {e}")
    print(f"Loaded {len(images)} images from {directory}")
    return np.array(images), np.array(labels)

# Load training and validation data (use a smaller subset for testing)
train_images, train_labels = load_images_and_labels(train_dir, max_images=1000)
test_images, test_labels = load_images_and_labels(validation_dir, max_images=1000)

# Convert images to PyTorch tensors and permute dimensions
train_images = torch.tensor(train_images, dtype=torch.float32).permute(0, 3, 1, 2)
test_images = torch.tensor(test_images, dtype=torch.float32).permute(0, 3, 1, 2)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# Split training data into training and validation
train_size = int(0.8 * len(train_images))
val_size = len(train_images) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(TensorDataset(train_images, train_labels), [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Instantiate the model, loss function, and optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to train the model
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_loss = np.inf
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        val_loss /= len(val_loader)
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save the model if validation loss has decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    return train_losses, train_accs, val_losses, val_accs


# Train the model
train_losses, train_accs, val_losses, val_accs = train_model(model, criterion, optimizer, train_loader, val_loader,    num_epochs=10)

# Plot the error and accuracy curves
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Reload the best model
best_model = CNN()
best_model.load_state_dict(torch.load('best_model.pth'))

# Create DataLoader instance for testing
test_dataset = TensorDataset(test_images, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Function to test the model
def test_model(model, criterion, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_predicted.extend(predicted.tolist())
            all_labels.extend(labels.tolist())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss /= len(test_loader)
    test_acc = correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    return all_predicted, all_labels


# Test the model
predicted_labels, true_labels = test_model(best_model, criterion, test_loader)

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate F1 score
f1_cnn = f1_score(true_labels, predicted_labels, average='weighted')
print(f"Average F1 Score: {f1_cnn:.4f}")

# Compare the results
print("Comparison of Models:")
print(f"SVM F1 Score: {f1_svm}")
print(f"Neural Network F1 Score: {f1_nn}")
print(f"CNN F1 Score: {f1_cnn}")

best_model = "SVM" if f1_svm > max(f1_nn, f1_cnn) else "Neural Network" if f1_nn > f1_cnn else "CNN"
print(f"The best model is: {best_model}")