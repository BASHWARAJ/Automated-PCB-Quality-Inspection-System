import torch
from torch.utils.data import DataLoader
from dataset import PCBDefectDataset
from model import PCBDefectDetector
from pathlib import Path
import os

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================= PATHS =================
repo_root = Path(__file__).resolve().parent.parent
train_img_dir = repo_root / "dataset" / "train" / "images"
train_label_dir = repo_root / "dataset" / "train" / "labels"

# ================= SETTINGS =================
NUM_CLASSES = 6 + 1  # original classes + background
MAX_OBJECTS = 10

# ================= DATASET =================
train_ds = PCBDefectDataset(
    img_dir=str(train_img_dir),
    label_dir=str(train_label_dir),
    img_size=256,
    num_classes=NUM_CLASSES,
    max_objects=MAX_OBJECTS
)

train_loader = DataLoader(
    train_ds,
    batch_size=5,
    shuffle=True,
    drop_last=True
)

# ================= MODEL =================
model = PCBDefectDetector(num_classes=NUM_CLASSES, num_preds=MAX_OBJECTS).to(device)

# ================= LOSSES =================
# IMPORTANT: Use plain CE first (no class weights while debugging)
cls_loss_fn = torch.nn.CrossEntropyLoss()
bbox_loss_fn = torch.nn.SmoothL1Loss()

# Lower LR for from-scratch detector
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 20
BBOX_LOSS_WEIGHT = 2.0

# ================= TRAINING =================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for imgs, labels, boxes in train_loader:
        imgs = imgs.to(device)
        labels = labels.long().to(device)
        boxes = boxes.to(device)

        optimizer.zero_grad()

        cls_pred, box_pred = model(imgs)  # cls_pred: (B, K, C), box_pred: (B, K, 4)

        # classification loss over all slots
        loss_cls = cls_loss_fn(cls_pred.view(-1, NUM_CLASSES), labels.view(-1))

        # bbox loss only for positive slots (labels != background)
        pos_mask = labels != (NUM_CLASSES - 1)
        if pos_mask.sum() > 0:
            pred_boxes_pos = box_pred[pos_mask]
            target_boxes_pos = boxes[pos_mask]
            loss_box = bbox_loss_fn(pred_boxes_pos, target_boxes_pos)
        else:
            loss_box = torch.tensor(0.0, device=device)

        loss = loss_cls + BBOX_LOSS_WEIGHT * loss_box
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Avg Loss: {avg_loss:.4f}"
    )

# ================= SAVE MODEL =================
outputs_dir = repo_root / "outputs"
os.makedirs(outputs_dir, exist_ok=True)

model_path = outputs_dir / "pcb_defect_model.pth"
torch.save(model.state_dict(), model_path)

print(f"\n✅ Model saved at: {model_path}")
