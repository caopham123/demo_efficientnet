import timm
import torch
import torch.nn as nn
import torch.optim as optim     # For optimizer
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader     # For iterating through data

# from timm.data import resolve_data_config
# from timm.data.transforms_factory import create_transform
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load and Prepare Data
def prepare_data(data_dir='./dataset', batch_size=32):
    # For this example, we'll use CIFAR-10 (change to your dataset)
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),  # EfficientNetV2 expects at least 224x224
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    
    val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    num_classes = len(train_dataset.classes)
    
    return train_loader, val_loader, num_classes

# Step 2: Create Model
def create_model(num_classes=10, pretrained=True):
    # List available EfficientNetV2 models in timm
    # print("Available EfficientNetV2 models in timm:")
    # for model_name in timm.list_models('efficientnetv2*'):
    #     print(f" - {model_name}")
    
    model_name = 'efficientnetv2_rw_s'
    
    # Create model
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model.to(device)

# Step 3: Training Loop
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Track metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        ## Training loop with progress bar
        train_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for inputs, labels in train_iter:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            train_iter.set_postfix({
                'loss': running_loss/(train_iter.n+1),
                'acc': 100.*correct/total
            })
        
        # Update learning rate
        scheduler.step()
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        ## Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for inputs, labels in val_iter:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                val_iter.set_postfix({
                    'loss': val_loss/(val_iter.n+1),
                    'acc': 100.*correct/total
                })
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'===> Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return train_losses, val_losses, train_accs, val_accs

# Step 4: Visualization
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

# Step 5: Inference Example
def inference_example(model, class_names):
    """Run inference on a single example"""
    # Get a sample from validation set
    sample, label = next(iter(val_loader))
    sample, label = sample[0].unsqueeze(0).to(device), label[0].to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        output = model(sample)
        _, predicted = output.max(1)
        probabilities = torch.nn.functional.softmax(output, dim=1)
    
    # Display
    sample = sample.squeeze().cpu().permute(1, 2, 0).numpy()
    sample = sample * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Un-normalize
    sample = sample.clip(0, 1)
    
    plt.imshow(sample)
    plt.title(f'True: {class_names[label]}\nPredicted: {class_names[predicted]} ({probabilities[0][predicted].item():.2f})')
    plt.axis('off')
    plt.show()

# Main execution
if __name__ == '__main__':
    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.001
    
    # Prepare data
    train_loader, val_loader, num_classes = prepare_data(batch_size=BATCH_SIZE)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']  # CIFAR-10 classes
    
    # Create model
    model = create_model(num_classes=num_classes, pretrained=True)
    
    # Train model
    print("\nStarting training...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, 
        epochs=EPOCHS, lr=LEARNING_RATE
    )
    
    # Plot metrics
    plot_metrics(train_losses, val_losses, train_accs, val_accs)
    
    # Example inference
    print("\nRunning inference example...")
    inference_example(model, class_names)