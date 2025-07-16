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

if not os.path.exists(CHECKPOINT_DIR):
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

    return (DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False),
            train_ds.class_to_idx)

def create_model(num_classes):
    try:
        model = timm.create_model(
            MODEL_NAME, pretrained=True, num_classes= num_classes)
        return model.to(DEVICE)
    except Exception as e:
        print(f"Err loading model:{e}")

def one_hot_encoded(labels, num_class):
    return F.one_hot(labels, num_class).float()

def save_checkpoint(state, filename):
    torch.save(state, os.path.join(CHECKPOINT_DIR, filename))
    print(f"Checkpoint saved to {filename}")

def train_model(train_loader, val_loader, num_classes, epochs=EPOCH_NUM):
    model = create_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # Track metrics
    best_val_acc = 0.0  # Track best validation accuracy
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
    }

    for epoch in range (epochs):
        # Training phase:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

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
                'acc': 100.0 * (correct/total),
            })
        # Update learning rate
        scheduler.step()

        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * (correct / total)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        ## Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

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
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'===> Epoch {epoch+1}/{epochs}: '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # ===== MODEL CHECKPOINTING =====
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'history': history,
            'class_to_idx': cls_to_idx
        }, filename="last.pt")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'history': history,
                'class_to_idx': cls_to_idx,
                'best_val_acc': best_val_acc,
            }, filename = 'best.pt')
            print(f"New best model saved with val_acc: {val_acc:.2f}%")
        # ==============================

    # # Save final model after training completed
    # final_model_path = os.path.join(CHECKPOINT_DIR, 'final.pt')
    # torch.save({
    #     'state_dict': model.state_dict(),
    #     'class_to_idx': cls_to_idx,
    #     'history': history
    # }, final_model_path)
    # print(f"Final model saved to {final_model_path}")
    return history
    # return train_losses, val_losses, train_accs, val_accs

def load_checkpoint(filename="best.pt"):
    """Load a training checkpoint"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, filename)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    model = create_model(num_classes=len(checkpoint['class_to_idx'])).to(DEVICE)
    model.load_state_dict(checkpoint['state_dict'])

    # ========== Restore optimizer and scheduler if present
    if 'optimizer' in checkpoint and 'scheduler' in checkpoint:
        optimizer = optim.AdamW(model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH_NUM)
        scheduler.load_state_dict(checkpoint['scheduler'])
        return model, checkpoint, optimizer, scheduler
    # =========================
    return model, checkpoint

def plot_metrics(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(18, 6))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    
    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy over Epochs')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'training_metrics.png'))
    plt.show()

if __name__ == "__main__":
    train_loader, val_loader, cls_to_idx = get_data_loader()
    num_classes =len(cls_to_idx)

    # Train the model
    history = train_model(train_loader, val_loader, num_classes)
    
    # Plot and save metrics
    plot_metrics(history)
    print("\nTraining completed!")
    print(f"Best model saved to: {os.path.join(CHECKPOINT_DIR, 'best.pt')}")
    print(f"Last model saved to: {os.path.join(CHECKPOINT_DIR, 'last.pt')}")
