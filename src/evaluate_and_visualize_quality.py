import os
import time
import cv2
import torch
import numpy as np
import argparse
import requests
import io
from urllib.parse import urlparse
from model import PCBDefectDetector
from dataset import PCBDefectDataset
from pathlib import Path

# ================= CONFIG =================
CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper"
]

IMG_SIZE = 256
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.25
MAX_OBJECTS = 10

# Resolve paths relative to repository root so the script works from src/ or project root
repo_root = Path(__file__).resolve().parent.parent
MODEL_PATH = str(repo_root / "outputs" / "pcb_defect_model.pth")

IMG_DIR = str(repo_root / "dataset" / "test" / "images")
LABEL_DIR = str(repo_root / "dataset" / "test" / "labels")

WINDOW_NAME = "PCB Quality Inspection"

# ================= ARGPARSE =================
parser = argparse.ArgumentParser(description="PCB quality inspection - multi-source input")
parser.add_argument("--source", "-s", default="dataset",
                    help="Input source: 'dataset' (default) to run evaluation on test set, or path/URL/rtsp/video/image to run inference")
parser.add_argument("--save", action="store_true", help="Save output for video inputs to 'outputs/output.mp4'")
parser.add_argument("--conf", type=float, default=0.05, help="Confidence threshold for displaying detections (default: 0.05)")
parser.add_argument("--debug", action="store_true", help="Enable debug logging for detections per image")
args = parser.parse_args()

# apply CLI flags
CONF_THRESHOLD = args.conf
DEBUG = args.debug

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================= DATASET =================
NUM_CLASSES = len(CLASS_NAMES) + 1  # add background class as last index

dataset = PCBDefectDataset(
    img_dir=IMG_DIR,
    label_dir=LABEL_DIR,
    img_size=IMG_SIZE,
    num_classes=NUM_CLASSES,
    max_objects=MAX_OBJECTS
)

# ================= MODEL =================
model = PCBDefectDetector(num_classes=NUM_CLASSES, num_preds=MAX_OBJECTS)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except Exception as e:
    print(f"Warning: failed to load model weights ({e}). You may need to retrain the model to match new architecture.")
model.to(device)
model.eval()

# ================= IOU =================
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / (area1 + area2 - inter + 1e-6)

# ================= METRICS =================
tp = fp = fn = 0
inference_times = []

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)


def is_url(path):
    try:
        result = urlparse(path)
        return result.scheme in ("http", "https", "rtsp")
    except Exception:
        return False


def read_image_from_url(url):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    buf = io.BytesIO(resp.content)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def preprocess_frame(frame, img_size=IMG_SIZE):
    img = cv2.resize(frame, (img_size, img_size))
    tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)


def draw_prediction(orig, pred_boxes, pred_classes, confidences):
    # pred_* are lists (or arrays) of detections
    for pred_box, pred_cls, confidence in zip(pred_boxes, pred_classes, confidences):
        h, w, _ = orig.shape
        x1 = int(pred_box[0] * w)
        y1 = int(pred_box[1] * h)
        x2 = int(pred_box[2] * w)
        y2 = int(pred_box[3] * h)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        area = abs(x2 - x1) * abs(y2 - y1)
        if area < 1000:
            severity = "Low"
        elif area < 5000:
            severity = "Medium"
        else:
            severity = "High"

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        label = f"{CLASS_NAMES[pred_cls]} | {confidence:.2f} | {severity}"

        cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(orig, (center_x, center_y), 4, (0, 0, 255), -1)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = y1 - 10 if y1 - 10 > 10 else y1 + th + 10

        cv2.rectangle(orig, (x1, ty - th - 5), (x1 + tw + 6, ty), (0, 255, 0), -1)
        cv2.putText(orig, label, (x1 + 3, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return orig


def run_inference_on_frame(frame):
    inp = preprocess_frame(frame)
    with torch.no_grad():
        start = time.time()
        cls_pred, box_pred = model(inp)  # cls_pred: (1, K, C), box_pred: (1, K, 4)
        inference_times.append(time.time() - start)

    probs = torch.softmax(cls_pred, dim=2)[0]  # (K, C)
    boxes = box_pred[0].cpu().numpy()  # (K,4)

    confidences, classes = probs.max(dim=1)
    confidences = confidences.cpu().numpy()
    classes = classes.cpu().numpy()

    # filter out background and low-confidence
    preds = []
    for b, c, conf in zip(boxes, classes, confidences):
        if c == (NUM_CLASSES - 1):
            continue
        if conf < CONF_THRESHOLD:
            continue
        preds.append((b, int(c), float(conf)))

    if len(preds) == 0:
        return [], [], []

    # sort by confidence desc
    preds.sort(key=lambda x: x[2], reverse=True)
    boxes_out = [p[0] for p in preds]
    classes_out = [p[1] for p in preds]
    confs_out = [p[2] for p in preds]
    return boxes_out, classes_out, confs_out


def handle_image_path(path):
    if is_url(path):
        orig = read_image_from_url(path)
    else:
        orig = cv2.imread(path)
    if orig is None:
        print(f"Failed to load image: {path}")
        return

    boxes, classes, confs = run_inference_on_frame(orig)
    if DEBUG:
        print(f"[DEBUG] detections: {len(boxes)}")
    if len(boxes) == 0:
        print("No detections")
    out = draw_prediction(orig, boxes, classes, confs)
    cv2.imshow(WINDOW_NAME, out)
    for c, conf in zip(classes, confs):
        print("Defect Type :", CLASS_NAMES[c])
        print("Confidence  :", round(conf, 2))
    key = cv2.waitKey(0)
    if key == ord("q"):
        cv2.destroyAllWindows()


def handle_video_source(source, save=False):
    cap = cv2.VideoCapture(source)
    writer = None
    if save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        boxes, classes, confs = run_inference_on_frame(frame)
        if DEBUG:
            print(f"[DEBUG] frame detections: {len(boxes)}")
        out = draw_prediction(frame, boxes, classes, confs)
        cv2.imshow(WINDOW_NAME, out)

        if writer is None and save:
            h, w, _ = out.shape
            writer = cv2.VideoWriter(str(repo_root / "outputs" / "output.mp4"), fourcc, 20.0, (w, h))
        if writer is not None:
            writer.write(out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if args.source == "dataset":
    # ================= DATASET =================
    with torch.no_grad():
        for idx in range(len(dataset)):
            img_tensor, gt_labels, gt_boxes = dataset[idx]

            img_name = dataset.images[idx]
            img_path = os.path.join(IMG_DIR, img_name)
            orig = cv2.imread(img_path)
            h, w, _ = orig.shape

            # run model on original image
            boxes_pred, classes_pred, confs_pred = run_inference_on_frame(orig)

            # prepare GT lists (only non-background)
            gt_labels = gt_labels.numpy()
            gt_boxes = gt_boxes.numpy()
            gt_list = []
            for g_c, g_b in zip(gt_labels, gt_boxes):
                if g_c == (NUM_CLASSES - 1):
                    continue
                gt_list.append((int(g_c), g_b))

            matched_gt = [False] * len(gt_list)

            # match predictions to GT greedily by best IoU (class-aware)
            for pred_b, pred_c, pred_conf in zip(boxes_pred, classes_pred, confs_pred):
                best_i = -1
                best_iou = 0.0
                for i, (g_c, g_b) in enumerate(gt_list):
                    if matched_gt[i]:
                        continue
                    if g_c != pred_c:
                        continue
                    iou = compute_iou(pred_b, g_b)
                    if iou > best_iou:
                        best_iou = iou
                        best_i = i

                if best_i >= 0 and best_iou >= IOU_THRESHOLD:
                    tp += 1
                    matched_gt[best_i] = True
                else:
                    fp += 1

            # remaining unmatched GT are false negatives
            fn += matched_gt.count(False)

            out = draw_prediction(orig, boxes_pred, classes_pred, confs_pred)
            cv2.imshow(WINDOW_NAME, out)

            for c, conf in zip(classes_pred, confs_pred):
                print("Defect Type :", CLASS_NAMES[c])
                print("Confidence  :", round(conf, 2))
            print("-" * 40)

            key = cv2.waitKey(0)
            if key == ord("q"):
                break

    cv2.destroyAllWindows()

else:
    src = args.source
    # If source is an image URL or ends with image extension -> single image
    if is_url(src) and not src.lower().endswith(('.mp4', '.avi', '.mkv')):
        handle_image_path(src)
    elif os.path.isfile(src) and src.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
        handle_image_path(src)
    else:
        # treat as video file, camera index, or rtsp stream
        # allow numeric camera index
        try:
            idx = int(src)
            handle_video_source(idx, save=args.save)
        except Exception:
            handle_video_source(src, save=args.save)

# ================= FINAL METRICS =================
precision = tp / (tp + fp + 1e-6)
recall = tp / (tp + fn + 1e-6)
map_50 = precision * recall
fps = 1 / (sum(inference_times) / len(inference_times))
model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)

print("\n===== FINAL QUALITY INSPECTION EVALUATION =====")
print(f"mAP@0.5    : {map_50:.4f}")
print(f"FPS        : {fps:.2f}")
print(f"Model Size : {model_size:.2f} MB")
print("==============================================")
