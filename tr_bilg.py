import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# Klasörlerin tanımlanması
data_path = Path('Data')
folders = ['glioma_tumor', 'meningioma_tumor', 'normal', 'pituitary_tumor']
img_size = (128, 128)

# data ve etiket listeleri
data = []
labels = []

# Her klasörde dolaşma
for folder_idx, folder in enumerate(folders):
    folder_path = data_path / folder
    
    # Klasördeki tüm fotoğrafları al
    for filename in tqdm(os.listdir(folder_path), desc=f"Processing {folder}"):
        file_path = folder_path / filename
        
        # Fotoğrafı yükleme
        img = cv2.imread(str(file_path))
        
        # Dosyanın başarılı bir şekilde yüklenip yüklenmediğini kontrol etme
        if img is None:
            print(f"Warning: {file_path} could not be loaded.")
            continue
        
        # Boyutlandırma ve renk kanallarını düzeltme
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Veri listesine ekleme
        data.append(img)
        
        # Etiket listesine ekleme
        labels.append(folder_idx)

# Veri ve etiket listelerini numpy dizilerine dönüştürme
data = np.array(data)
labels = np.array(labels)

# Veri ve etiketlerin şeklini kontrol etme
print("Veri şekli:", data.shape)
print("Etiket şekli:", labels.shape)

# Veriyi train ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)

# Verileri normalize etme
X_train = X_train / 255.0
X_test = X_test / 255.0

# Verileri tensor formatına dönüştürme
X_train = torch.tensor(X_train.transpose((0, 3, 1, 2))).float()
X_test = torch.tensor(X_test.transpose((0, 3, 1, 2))).float()
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Train ve test veri setleri oluşturma
class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

# Transformlar
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # ResNet için 224x224 boyutuna getiriyoruz
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset ve DataLoader oluşturma
train_dataset = ImageDataset(X_train, y_train, transform=train_transforms)
test_dataset = ImageDataset(X_test, y_test, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model tanımlama ve yüklenmesi (ResNet18 kullanıyoruz)
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # 4 classes

# Optimizer ve loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# DataLoader üzerinden veri dolaşma ve model eğitimi
num_epochs = 40
train_losses = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    
    train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    
    # Test seti üzerinde modelin performansını test etme
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels) 
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    test_loss = test_loss / len(test_loader.dataset)
    test_losses.append(test_loss)
    
    # Accuracy hesaplama
    test_accuracy = correct / total
    test_accuracies.append(test_accuracy)
    
    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {train_loss:.4f}, "
          f"Test Loss: {test_loss:.4f}, "
          f"Test Accuracy: {test_accuracy:.4f}")

# Training loss grafiği
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(test_losses, label='Test Loss', marker='x')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Accuracy grafiği
plt.figure(figsize=(10, 5))
plt.plot(test_accuracies, label='Test Accuracy', marker='o', color='r')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Son modeli kaydetme
torch.save(model.state_dict(), 'resnet_model.pth')

# Son durumdaki loss ve accuracy değerleri
print(f"Final Test Loss: {test_losses[-1]:.4f}, Final Test Accuracy: {test_accuracies[-1]:.4f}")
