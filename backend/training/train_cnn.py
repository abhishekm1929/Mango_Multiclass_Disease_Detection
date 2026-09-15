"""
EfficientNet-B0 Transfer Learning Training Pipeline for Mango Leaf Disease Classification.
Executes fine-tuning with Cross-Entropy loss, AdamW optimizer, Cosine Annealing, and checkpointing.
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys

# Ensure backend directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "training"))

from dataset import create_dataloaders, MANGO_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
from classifier import build_efficientnet_classifier


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=1, total_epochs=6):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_t = time.time()

    total_batches = len(dataloader)
    for batch_idx, (images, labels) in enumerate(dataloader, 1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 15 == 0 or batch_idx == total_batches:
            batch_loss = running_loss / max(1, total)
            batch_acc = (correct / max(1, total)) * 100.0
            elapsed = time.time() - start_t
            speed = total / max(0.1, elapsed)
            print(f"  [Epoch {epoch:02d}/{total_epochs:02d}] Batch {batch_idx:02d}/{total_batches:02d} - Loss: {batch_loss:.4f}, Acc: {batch_acc:.1f}% ({speed:.1f} img/s)", flush=True)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = (correct / max(1, total)) * 100.0
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    class_correct = {i: 0 for i in range(len(MANGO_CLASSES))}
    class_total = {i: 0 for i in range(len(MANGO_CLASSES))}

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for p, l in zip(preds, labels):
                l_item = l.item()
                class_total[l_item] += 1
                if p.item() == l_item:
                    class_correct[l_item] += 1

    val_loss = running_loss / max(1, total)
    val_acc = (correct / max(1, total)) * 100.0

    class_accs = {}
    for i in range(len(MANGO_CLASSES)):
        c_tot = class_total[i]
        c_cor = class_correct[i]
        class_accs[IDX_TO_CLASS[i]] = (c_cor / max(1, c_tot)) * 100.0 if c_tot > 0 else 0.0

    return val_loss, val_acc, class_accs


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 4))

    default_data_dir = os.path.join(BASE_DIR, "data", "Mango S data")
    default_save_path = os.path.join(BASE_DIR, "models", "mango_cnn_efficientnet.pth")

    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on Mango Leaf Disease Dataset")
    parser.add_argument("--data_dir", type=str, default=default_data_dir, help="Path to dataset root folder")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=6e-4, help="Initial learning rate")
    parser.add_argument("--max_samples", type=int, default=250, help="Max images per class (default 250 for fast high-accuracy training)")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="L2 weight decay regularization")
    parser.add_argument("--save_path", type=str, default=default_save_path, help="Output weights file path")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 65, flush=True)
    print("EfficientNet-B0 High-Precision Mango Leaf Disease Training Engine", flush=True)
    print(f"Device: {device} | CPU Threads: {torch.get_num_threads()} | Classes: {len(MANGO_CLASSES)}", flush=True)
    print(f"Data directory: {args.data_dir}", flush=True)
    print(f"Max samples per class: {args.max_samples}", flush=True)
    print(f"Save weights to: {args.save_path}", flush=True)
    print("=" * 65, flush=True)

    if not os.path.exists(args.data_dir):
        print(f"Error: Dataset directory '{args.data_dir}' not found.", flush=True)
        return

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        val_ratio=0.15,
        test_ratio=0.10,
        max_samples_per_class=args.max_samples,
        seed=42
    )

    if len(train_loader.dataset) == 0:
        print("Error: No training samples found. Please check dataset folder structure.", flush=True)
        return

    print(f"Dataset split -> Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}", flush=True)
    print(f"Class mapping: {CLASS_TO_IDX}\n", flush=True)

    # Build model with pretrained backbone
    model = build_efficientnet_classifier(num_classes=len(MANGO_CLASSES), pretrained=True)
    
    # Freeze lower feature blocks (0 to 4) for faster, stable transfer learning
    for param in model.features[:5].parameters():
        param.requires_grad = False

    model = model.to(device)

    # Trainable parameters: top feature blocks (5 to 8) + classifier head
    trainable_features = [p for p in model.features[5:].parameters() if p.requires_grad]
    classifier_params = list(model.classifier.parameters())

    optimizer = optim.AdamW([
        {"params": trainable_features, "lr": args.lr * 0.25},
        {"params": classifier_params, "lr": args.lr}
    ], weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Unfreeze all layers in the last 2 epochs for fine polishing
        if epoch == args.epochs - 1:
            for param in model.features[:5].parameters():
                param.requires_grad = True

        start_t = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=epoch, total_epochs=args.epochs)
        val_loss, val_acc, class_accs = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - start_t

        print(f"\n--- Epoch [{epoch:02d}/{args.epochs:02d}] Summary ({elapsed:.1f}s) ---", flush=True)
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%", flush=True)
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%", flush=True)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch,
                "model_name": "EfficientNet-B0",
                "state_dict": model.state_dict(),
                "classes": MANGO_CLASSES,
                "class_to_idx": CLASS_TO_IDX,
                "best_val_acc": best_val_acc,
                "class_accs": class_accs
            }
            torch.save(checkpoint, args.save_path)
            print(f"  --> Saved new best checkpoint to '{args.save_path}' (Val Acc: {best_val_acc:.2f}%)", flush=True)
            for c_name, c_acc in class_accs.items():
                print(f"      - {c_name:<18}: {c_acc:.1f}%", flush=True)
        print("-" * 65, flush=True)

    # Final evaluation on hold-out test set
    if len(test_loader.dataset) > 0 and os.path.exists(args.save_path):
        print("\n=== Final Hold-out Test Set Evaluation ===", flush=True)
        best_ckpt = torch.load(args.save_path, map_location=device)
        model.load_state_dict(best_ckpt["state_dict"])
        test_loss, test_acc, test_class_accs = validate(model, test_loader, criterion, device)
        print(f"Final Test Accuracy: {test_acc:.2f}% (Loss: {test_loss:.4f})", flush=True)
        for c_name, c_acc in test_class_accs.items():
            print(f"  * {c_name:<18}: {c_acc:.1f}%", flush=True)

    print(f"\nTraining completed! Best Validation Accuracy: {best_val_acc:.2f}%. Model saved to '{args.save_path}'.", flush=True)


if __name__ == "__main__":
    main()
