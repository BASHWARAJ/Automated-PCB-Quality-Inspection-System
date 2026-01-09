import torch
import torch.nn as nn


class PCBDefectDetector(nn.Module):
    def __init__(self, num_classes: int, num_preds: int = 10):
        """A simple fixed-slot detector that predicts `num_preds` boxes per image.

        - `num_classes` should include a background class (last index).
        - `num_preds` is the maximum number of detections predicted per image.
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_preds = num_preds

        # -------- Backbone --------
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        # -------- Shared feature layer --------
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True)
        )

        # -------- Heads (predict `num_preds` slots) --------
        self.cls_head = nn.Linear(128, self.num_preds * self.num_classes)
        self.box_head = nn.Linear(128, self.num_preds * 4)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)

        cls_logits = self.cls_head(x)
        box_out = self.box_head(x)

        # reshape to (batch, num_preds, ...)
        batch = cls_logits.shape[0]
        cls_logits = cls_logits.view(batch, self.num_preds, self.num_classes)
        # bbox normalized to [0,1]
        bbox = torch.sigmoid(box_out).view(batch, self.num_preds, 4)

        return cls_logits, bbox
