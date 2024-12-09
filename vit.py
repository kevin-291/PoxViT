import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

# Define hyperparameters
IMG_SIZE = 224
NUM_CLASSES = 6
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transform function
def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


# Define data loading function
def load_data(data_dir):
    transform = get_transforms()
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader

# Define the Vision Transformer model with attention map functionality
class SkinDiseaseVisionTransformer(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SkinDiseaseVisionTransformer, self).__init__()
        self.vit = models.vit_b_16(weights='IMAGENET1K_V1')
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)

    def get_attention_maps(self):
        return self.attention_maps

# Attention map visualization function
def visualize_attention_map(model, image, label, head_idx=0):
    model.eval()
    with torch.no_grad():
        image = image.to(DEVICE).unsqueeze(0)
        _ = model(image)
        attention_map = model.get_attention_map()[head_idx].squeeze(0).cpu().numpy()
    
    attention_map_resized = cv2.resize(attention_map, (IMG_SIZE, IMG_SIZE))
    attention_map_resized = np.maximum(attention_map_resized, 0)
    attention_map_resized /= attention_map_resized.max()
    
    image_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 0.5 + 0.5)
    plt.imshow(image_np)
    plt.imshow(attention_map_resized, cmap='jet', alpha=0.5)
    plt.title(f"Attention Map (Head {head_idx}) for Label: {label}")
    plt.axis('off')
    plt.show()

# Training function
def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    with tqdm(loader, desc="Training", leave=False) as pbar:
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})
    return running_loss / len(loader)

# Validation function
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad(), tqdm(loader, desc="Validation", leave=False) as pbar:
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            accuracy = 100 * correct / total
            pbar.set_postfix({"Loss": loss.item(), "Accuracy": accuracy})
    
    accuracy = 100 * correct / total
    return running_loss / len(loader), accuracy

# Testing function
def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(), tqdm(loader, desc="Testing") as pbar:
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total
    return accuracy

# Save model function
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    logging.info(f"Model saved to {filepath}")

# Load model function
def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.to(DEVICE)
    model.eval()
    logging.info(f"Model loaded from {filepath}")

# Main function for training, validating, testing, and saving the model
def main(data_dir):
    logging.info("Loading data...")
    train_loader, val_loader, test_loader = load_data(data_dir)
    
    logging.info("Initializing model, criterion, and optimizer...")
    model = SkinDiseaseVisionTransformer().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses, val_losses, val_accuracies = [], [], []
    for epoch in range(EPOCHS):
        logging.info(f"Epoch [{epoch+1}/{EPOCHS}] starting...")
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = validate(model, val_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        logging.info(f"Epoch [{epoch+1}/{EPOCHS}] completed - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    
    logging.info("Starting testing phase...")
    test_accuracy = test(model, test_loader)
    logging.info(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Save the trained model
    save_model(model, "skin_disease_vit.pth")
    
    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    
    # Plot validation accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.show()

    # Visualize attention maps for the first few images in the test set
    examples = iter(test_loader)
    example_images, example_labels = next(examples)
    for i in range(3):  # Show attention maps for first 3 images
        visualize_attention_map(model, example_images[i], example_labels[i].item())

# Run main with the path to the dataset
data_dir = 'sorted_dataset'  # Update with your dataset path
main(data_dir)
