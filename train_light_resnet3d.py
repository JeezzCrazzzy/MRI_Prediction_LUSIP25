import torch
import torch.nn as nn
import torch.optim as optim
import time
import psutil
import GPUtil
from mri_dataloader import get_dataloader
from light_resnet3d import LightResNet3D

# --- System Info ---
print("=== SYSTEM INFORMATION ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"CPU cores: {psutil.cpu_count()}")
print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
print("=" * 50)

# --- Hyperparameters ---
batch_size = 1
num_epochs = 10
learning_rate = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Data Loading ---
print("\n=== DATA LOADING ===")
print("Loading training data...")
start_time = time.time()
train_loader, class_to_idx = get_dataloader('train.csv', batch_size=batch_size, shuffle=True)
data_load_time = time.time() - start_time
print(f"Training data loaded in {data_load_time:.2f} seconds")
print(f"Number of training batches: {len(train_loader)}")

print("Loading validation data...")
start_time = time.time()
val_loader, _ = get_dataloader('val.csv', batch_size=batch_size, shuffle=False)
data_load_time = time.time() - start_time
print(f"Validation data loaded in {data_load_time:.2f} seconds")
print(f"Number of validation batches: {len(val_loader)}")

num_classes = len(class_to_idx)
print(f"Number of classes: {num_classes}")
print(f"Class mapping: {class_to_idx}")

# Get sample to check input shape
print("\nChecking input data shape...")
sample_batch = next(iter(train_loader))
sample_img, sample_label = sample_batch
print(f"Input shape: {sample_img.shape}")  # Expected: [batch_size, channels, depth, height, width]
print(f"Input dtype: {sample_img.dtype}")
print(f"Label shape: {sample_label.shape}")
print(f"Sample label: {sample_label}")

# --- Model Setup ---
print("\n=== MODEL SETUP ===")
print("Initializing model...")
model = LightResNet3D(num_classes=num_classes).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Test forward pass
print("Testing forward pass...")
start_time = time.time()
with torch.no_grad():
    test_output = model(sample_img.to(device))
forward_time = time.time() - start_time
print(f"Forward pass completed in {forward_time:.2f} seconds")
print(f"Output shape: {test_output.shape}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(f"Loss function: {criterion}")
print(f"Optimizer: {optimizer}")

best_val_acc = 0.0

# --- Training Loop ---
print(f"\n=== STARTING TRAINING ===")
print(f"Estimated time per epoch: {forward_time * len(train_loader) / 60:.1f} minutes (forward pass only)")
training_start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch+1}/{num_epochs}")
    print(f"{'='*60}")
    
    # --- Training Phase ---
    print(f"Starting training phase...")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    train_phase_start = time.time()
    
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        batch_start_time = time.time()
        
        # Data transfer
        data_transfer_start = time.time()
        imgs, labels = imgs.to(device), labels.to(device)
        data_transfer_time = time.time() - data_transfer_start
        
        # Forward pass
        forward_start = time.time()
        optimizer.zero_grad()
        outputs = model(imgs)
        forward_time = time.time() - forward_start
        
        # Loss computation
        loss_start = time.time()
        loss = criterion(outputs, labels)
        loss_time = time.time() - loss_start
        
        # Backward pass
        backward_start = time.time()
        loss.backward()
        backward_time = time.time() - backward_start
        
        # Optimizer step
        optim_start = time.time()
        optimizer.step()
        optim_time = time.time() - optim_start
        
        # Statistics
        running_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        batch_time = time.time() - batch_start_time
        
        # Memory usage
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1e9
            gpu_memory_cached = torch.cuda.memory_reserved() / 1e9
        else:
            gpu_memory = gpu_memory_cached = 0
        
        # Detailed batch logging
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
            current_acc = correct / total if total > 0 else 0
            print(f"[TRAIN] Batch {batch_idx+1:3d}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | Acc: {current_acc:.4f} | "
                  f"Time: {batch_time:.1f}s")
            print(f"  ├── Data Transfer: {data_transfer_time:.3f}s | Forward: {forward_time:.3f}s | "
                  f"Backward: {backward_time:.3f}s | Optimizer: {optim_time:.3f}s")
            if torch.cuda.is_available():
                print(f"  └── GPU Memory: {gpu_memory:.2f}GB used, {gpu_memory_cached:.2f}GB cached")
        
        # Progress indicator for long batches
        elif (batch_idx + 1) % 1 == 0:
            elapsed = time.time() - train_phase_start
            eta = (elapsed / (batch_idx + 1)) * (len(train_loader) - batch_idx - 1)
            print(f"[TRAIN] Batch {batch_idx+1:3d}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | ETA: {eta/60:.1f}min | "
                  f"Batch Time: {batch_time:.1f}s")

    train_phase_time = time.time() - train_phase_start
    train_loss = running_loss / total
    train_acc = correct / total
    
    print(f"\n[TRAIN SUMMARY] Epoch {epoch+1}")
    print(f"├── Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
    print(f"└── Phase Time: {train_phase_time/60:.1f} minutes")

    # --- Validation Phase ---
    print(f"\nStarting validation phase...")
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    val_phase_start = time.time()
    
    for batch_idx, (imgs, labels) in enumerate(val_loader):
        batch_start_time = time.time()
        imgs, labels = imgs.to(device), labels.to(device)
        
        with torch.no_grad():
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        
        val_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        val_correct += (predicted == labels).sum().item()
        val_total += labels.size(0)
        
        batch_time = time.time() - batch_start_time
        
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(val_loader):
            current_acc = val_correct / val_total if val_total > 0 else 0
            print(f"[VAL] Batch {batch_idx+1:3d}/{len(val_loader)} | "
                  f"Loss: {loss.item():.4f} | Acc: {current_acc:.4f} | "
                  f"Time: {batch_time:.1f}s")

    val_phase_time = time.time() - val_phase_start
    val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    
    print(f"\n[VAL SUMMARY] Epoch {epoch+1}")
    print(f"├── Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")
    print(f"└── Phase Time: {val_phase_time/60:.1f} minutes")

    # --- Epoch Summary ---
    epoch_time = time.time() - epoch_start_time
    total_elapsed = time.time() - training_start_time
    remaining_epochs = num_epochs - (epoch + 1)
    eta_total = (total_elapsed / (epoch + 1)) * remaining_epochs
    
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch+1} COMPLETE")
    print(f"├── Train: Loss {train_loss:.4f}, Acc {train_acc:.4f}")
    print(f"├── Val:   Loss {val_loss:.4f}, Acc {val_acc:.4f}")
    print(f"├── Epoch Time: {epoch_time/60:.1f} minutes")
    print(f"├── Total Time: {total_elapsed/60:.1f} minutes")
    print(f"└── ETA: {eta_total/60:.1f} minutes ({remaining_epochs} epochs remaining)")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc
        }, 'best_light_resnet3d.pth')
        print(f"✓ NEW BEST MODEL SAVED! (Val Acc: {val_acc:.4f})")

print(f"\n{'='*60}")
print("TRAINING COMPLETE!")
total_time = time.time() - training_start_time
print(f"Total training time: {total_time/60:.1f} minutes")
print(f"Best validation accuracy: {best_val_acc:.4f}")
print(f"{'='*60}")