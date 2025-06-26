import timm
import torch
import torch.nn as nn
import torch.optim as optim     # For optimizer
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader     # For iterating through data
from tqdm import tqdm
from setting import *



class EfficientNetV2TransferLearner:

    def __init__(self, model_name=MODEL_NAME, num_classes=10, pretrained=True):
        try:
            self.model = timm.create_model(
                model_name, pretrained=True, num_classes= num_classes)
            
        except Exception as e:
            print(f"Err loading model:{e}")

        
        self.device = ["cuda" if torch.cuda.is_available() else "cpu"]
    
    def get_data_loader(self, data_dir=DATASET_PATH, batch_size= BATCH_SIZE):
        train_transformer = transforms.Compose([
            transforms.RandomResizedCrop(size=(IMG_SIZE),scale=(0.2,0.8)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225])
        ])

        val_transformer = transforms.Compose([
            transforms.Resize(size=IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225])
        ])

        train_ds = datasets.ImageFolder(f"{TRAIN_PATH}", train_transformer)
        val_ds = datasets.ImageFolder(f"{VAL_PATH}", val_transformer)

        return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                DataLoader(val_ds, batch_size=batch_size, shuffle=False))
    
    def train_model(self, train_loader, val_loader, epochs=EPOCH_NUM, lr=.001):
        device = self.device
        self.model = self.model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimer,T_max=epochs)
        # Track metrics
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        for epoch in range (epochs):
            # Training phase:
            self.model.train()
            running_loss = 0.0
            correct, total = 0

            train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for inputs, labels in train_iter:
                inputs, labels = inputs.to(device), labels.to(device)

                optimer.zero_grad()
                # Forward propagation
                outputs =self.model(inputs)
                loss = criterion(outputs, labels)
                # Backward
                loss.backward()
                optimer.step()
                
                # Track statistic
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update progress bar
                train_iter.set_postfix({
                    'loss': running_loss/(train_iter.n +1),
                    'acc': 100.0 * (correct/total)
                })

            # Update learning rate
            scheduler.step()

            # Calculate training metrics
            train_loss = running_loss / len(train_loader)
            train_acc = (correct/total) * 100.0
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            ## Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                for inputs, labels in val_iter:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                    val_iter.set_postfix({
                        "loss": val_loss/(val_iter.n + 1),
                        "acc": (correct/total) * 100.0
                    })

            # Calculate validation metrics
            val_loss = val_loss/ len(val_loader)
            val_acc = (correct/total) * 100.0
            val_accs.append(val_acc)
            val_losses.append(val_loss)
  
            print(f'===> Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
        return train_losses, val_losses, train_accs, val_accs
    
    def plot_metrics(train_losses, val_losses, train_accs, val_accs):
        """Plot training and validation metrics"""
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss over Epochs')
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Accuracy over Epochs')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":

    efficientNet = EfficientNetV2TransferLearner(MODEL_NAME)
    # Prepare data
    train_loader, val_loader, num_classes = efficientNet.get_data_loader()





