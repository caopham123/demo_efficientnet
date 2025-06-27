import timm
import torch
import torch.nn as nn
import torch.optim as optim     # For optimizer
from timm.data.auto_augment import rand_augment_transform
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader     # For iterating through data
from tqdm import tqdm
from setting import *
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Create directory for checkpoints if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def get_data_loader():
    train_transformer = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transformer = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(f"{TRAIN_PATH}", train_transformer)
    val_ds = datasets.ImageFolder(f"{VAL_PATH}", val_transformer)

    class_to_idx = train_ds.class_to_idx
    return (DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False),
            class_to_idx)

def one_hot_encoded(labels, num_class):
    return F.one_hot(labels, num_class).float()

def save_checkpoint(state, filename="checkpoint.pt"):
    torch.save(state, os.path.join(CHECKPOINT_DIR, filename))
    print(f"Checkpoint saved to {filename}")

def create_model(num_classes):
    try:
        model = timm.create_model(
            MODEL_NAME, pretrained=True, num_classes= num_classes)
        return model.to(DEVICE)
    except Exception as e:
        print(f"Err loading model:{e}")

def train_model(model, train_loader, val_loader, epochs=EPOCH_NUM, lr=.001):
    
    model = create_model(num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=EPOCH_NUM,
        pct_start=0.1
    )
    # Track metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0.0  # Track best validation accuracy

    for epoch in range (epochs):
        # Training phase:
        model.train()
        running_loss = 0.0
        correct, total = 0

        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, labels in train_iter:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            # Forward propagation
            outputs =model(inputs)
            loss = criterion(outputs, labels)
            # Backward
            loss.backward()
            optimizer.step()
            
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
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for inputs, labels in val_iter:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
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
        
        # ===== MODEL CHECKPOINTING =====
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'class_to_idx': cls_to_idx
        }, filename="last.pt")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, filename = 'best_model.pt')
            print(f"New best model saved with val_acc: {val_acc:.2f}%")
        # ==============================

        
    return train_losses, val_losses, train_accs, val_accs

def load_checkpoint(filename="best.pt"):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model = create_model(num_classes=len(checkpoint['class_to_idx']))
        model.load_state_dict(checkpoint['state_dict'])
        model.to(DEVICE)
        return model, checkpoint
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

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
    train_loader, val_loader, cls_to_idx = get_data_loader()
    num_classes =len(cls_to_idx)

    # Train the model
    train_losses, val_losses, train_accs, val_accs = train_model(
        train_loader, val_loader, num_classes)
    
    # Plot and save metrics
    plot_metrics(train_losses, val_losses, train_accs, val_accs)
