import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, datasets
from torchvision.transforms import functional as TF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
PATIENCE = 5
EARLY_STOPPING_DELTA = 0.001
MODEL_DIR = 'Model'
# Update paths
csv_path = '../archive/sample_labels.csv'
image_dir = '../archive/sample/images'

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ChestXRayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        super().__init__()
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.FloatTensor(label)

def load_and_preprocess_data(csv_path, image_dir):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        
        # Verify required columns exist
        required_columns = ['Image Index', 'Finding Labels']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV file must contain columns: {required_columns}")
        
        # Process labels
        df['Finding Labels'] = df['Finding Labels'].apply(lambda x: x.split('|'))
        
        # Initialize MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(df['Finding Labels'])
        
        # Create image paths and verify they exist
        image_paths = []
        valid_indices = []
        for idx, img_name in enumerate(df['Image Index']):
            img_path = os.path.join(image_dir, img_name)
            if os.path.exists(img_path):
                image_paths.append(img_path)
                valid_indices.append(idx)
            else:
                print(f"Warning: Image not found: {img_path}")
        
        # Filter labels to match valid images
        labels = labels[valid_indices]
        
        # Calculate class weights
        class_weights = {}
        for i in range(labels.shape[1]):
            class_weights[i] = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(labels[:, i]),
                y=labels[:, i]
            )[1]
        
        return image_paths, labels, mlb.classes_, class_weights
    except Exception as e:
        print(f"Error in load_and_preprocess_data: {str(e)}")
        raise

def create_data_loaders(image_paths, labels, test_size=0.2, val_size=0.1):
    try:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            image_paths, labels, test_size=test_size, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42
        )
        
        # Define transforms
        data_transforms = {
        'train': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomCrop((IMAGE_SIZE[0] - 10, IMAGE_SIZE[1] - 10)),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        }
        
        # Create datasets
        train_dataset = ChestXRayDataset(X_train, y_train, transform=data_transforms['train'])
        val_dataset = ChestXRayDataset(X_val, y_val, transform=data_transforms['val'])
        test_dataset = ChestXRayDataset(X_test, y_test, transform=data_transforms['val'])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        return train_loader, val_loader, test_loader, y_train, y_val, y_test
    except Exception as e:
        print(f"Error in create_data_loaders: {str(e)}")
        raise

class ChestXRayModel(nn.Module):
    def __init__(self, num_classes):
        super(ChestXRayModel, self).__init__()
        
        # Use EfficientNet-B4 as base model
        self.base_model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        
        # Modify the classifier
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512) ,
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Unfreeze last 20 layers
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.base_model.features[6:].parameters():
            param.requires_grad = True
        for param in self.base_model.classifier.parameters(): 
            param.requires_grad = True

    def forward(self, x):
        return self.base_model(x)

def train_model():
    try:
        # Load and preprocess data
        image_paths, labels, class_names, class_weights = load_and_preprocess_data(csv_path, image_dir)
        
        print(f"Loaded {len(image_paths)} images with {len(class_names)} classes")
        print(f"Class names: {class_names}")
        
        # Create data loaders
        train_loader, val_loader, test_loader, y_train, y_val, y_test = create_data_loaders(
            image_paths, labels
        )
        
        # Create model and move to GPU
        model = ChestXRayModel(len(class_names)).to(device)
        
        # Define loss function with class weights
        class_weights_tensor = torch.FloatTensor(list(class_weights.values())).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)
        
        # Define optimizer with weight decay
        optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=PATIENCE//2, verbose=True
        )
        
        # Early stopping
        best_val_loss = float('inf')
        early_stopping_counter = 0
        # Training loop
        for epoch in range(EPOCHS):
             # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]')
            for images, labels in progress_bar:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                train_correct += (predicted == labels).all(dim=1).sum().item()
                train_total += labels.size(0)
                
                # Update progress bar with current batch loss
                progress_bar.set_postfix({'loss': loss.item()})
            
            train_accuracy = train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    val_correct += (predicted == labels).all(dim=1).sum().item()
                    val_total += labels.size(0)
            
            val_accuracy = val_correct / val_total
            val_loss /= len(val_loader)
            
             # Update learning rate
            lr_scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss - EARLY_STOPPING_DELTA:
                best_val_loss = val_loss
                early_stopping_counter = 0
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pth'))
                print(f"Saved model to {os.path.join(MODEL_DIR, 'best_model.pth')}")
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= PATIENCE:
                    print(f"Early stopping triggered after {epoch+1} epochs. No improvement for {PATIENCE} epochs.")
                    break


            # Save best model
            
            
            
            
            
            print(f'\nEpoch {epoch+1}/{EPOCHS}:')
            print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # Load best model
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pth')))
        
        # Test phase
        model.eval()
        test_correct = 0
        test_total = 0
        
        y_true = []
        y_pred = []

        progress_bar = tqdm(test_loader, desc=f'Evaluating')
        with torch.no_grad():
            for images, labels in progress_bar:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits)
                predictions = (probs > 0.5).float()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Evaluate model
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        roc_auc = roc_auc_score(y_true, y_pred, average='macro')
        
        print(f'\nTest Accuracy: {accuracy:.4f}, Test Precision: {precision:.4f}, Test Recall: {recall:.4f}, Test F1-Score: {f1:.4f}, Test AUC-ROC: {roc_auc:.4f}')
        
         # Save final model and class names
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'chest_xray_model.pth'))
        np.save(os.path.join(MODEL_DIR, 'class_names.npy'), class_names)
        
        return model
    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        raise
if __name__ == "__main__":
    try:
        model = train_model()
    except Exception as e:
        print(f"Error during training: {str(e)}") 