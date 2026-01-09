import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class PCBDefectDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=256, num_classes=7, max_objects=10):
        """Dataset that returns up to `max_objects` labels and boxes per image.

        - `num_classes` must include a background class as the last index.
        - labels are returned as a tensor of shape (max_objects,), dtype long.
        - boxes are returned as a tensor of shape (max_objects, 4) in normalized xyxy format.
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.num_classes = num_classes
        self.max_objects = max_objects
        self.images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = cv2.imread(img_path)
        h, w, _ = image.shape
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0

        label_path = os.path.join(
            self.label_dir,
            img_name.replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt")
        )

        labels = np.full((self.max_objects,), fill_value=(self.num_classes - 1), dtype=np.int64)  # background
        boxes = np.zeros((self.max_objects, 4), dtype=np.float32)

        if os.path.exists(label_path):
            with open(label_path) as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]

            parsed = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    cls, xc, yc, bw, bh = map(float, parts[:5])
                    x1 = (xc - bw / 2)
                    y1 = (yc - bh / 2)
                    x2 = (xc + bw / 2)
                    y2 = (yc + bh / 2)
                    parsed.append((int(cls), [x1, y1, x2, y2]))

            # fill up to max_objects (truncate if too many)
            for i, (c, b) in enumerate(parsed[: self.max_objects]):
                labels[i] = c
                boxes[i] = np.array(b, dtype=np.float32)

        return image, torch.from_numpy(labels).long(), torch.from_numpy(boxes).float()
