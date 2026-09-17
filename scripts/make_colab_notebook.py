"""Generate notebooks/day5_auto_segmentation_colab.ipynb with SOTA Mask2Former & YOLO11x-seg."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "day5_auto_segmentation_colab.ipynb"

cells = []


def add_markdown(source: str):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()]
    })


def add_code(source: str):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()]
    })


# --- CELL 0: Markdown Intro ---
add_markdown("""# 🌟 Day 5 Image Segmentation Lab - SOTA Auto-Annotator (Mask2Former Swin-Large + YOLO11x-seg)

This Google Colab notebook runs the most advanced vision foundation models:
1. **Mask2Former (Swin-Large Backbone)**: SOTA for Semantic and Panoptic Segmentation on Cityscapes (`facebook/mask2former-swin-large-cityscapes-semantic` & `facebook/mask2former-swin-large-cityscapes-panoptic`).
2. **YOLO11x-seg**: Latest generation object and instance segmenter from Ultralytics.

---
### ⚙️ Quick Setup:
1. Go to **Runtime ➔ Change runtime type ➔ T4 GPU** (or A100 / V100 if available).
2. Run all cells from top to bottom (**Runtime ➔ Run all**).
3. The notebook will generate all 9 valid CVAT ZIP files and download `day5_all_submissions.zip`!
""")

# --- CELL 1: Code Setup ---
add_code("""# BƯỚC 0: Cài đặt thư viện Deep Learning SOTA
!pip install -q transformers torch torchvision ultralytics pycocotools opencv-python pillow scipy timm

import os
import io
import json
import zipfile
import shutil
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch

print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
""")

# --- CELL 2: Clone or Load Workspace ---
add_code("""# BƯỚC 1: Tải repo Day 5 vào môi trường Colab
import os
from pathlib import Path

REPO_DIR = Path("/content/Day5-Segmentation-Lab-Student")
if not REPO_DIR.exists():
    !git clone https://github.com/HoangAnhTuan123-git/Day5-Segmentation-Lab-Student.git {REPO_DIR}
    os.chdir(REPO_DIR)
else:
    os.chdir(REPO_DIR)

DATA_DIR = Path("data")
SUBMISSION_DIR = Path("submissions")
SUBMISSION_DIR.mkdir(exist_ok=True, parents=True)
print("Working directory:", os.getcwd())
""")

# --- CELL 3: Load SOTA Models ---
add_code("""# BƯỚC 2: Khởi tạo SOTA Models (Mask2Former Swin-Large + YOLO11x-seg)
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Đang tải Mask2Former Swin-Large Semantic...")
sem_model_name = "facebook/mask2former-swin-large-cityscapes-semantic"
sem_processor = AutoImageProcessor.from_pretrained(sem_model_name)
sem_model = Mask2FormerForUniversalSegmentation.from_pretrained(sem_model_name).to(device)
sem_model.eval()

print("Đang tải Mask2Former Swin-Large Panoptic...")
pan_model_name = "facebook/mask2former-swin-large-cityscapes-panoptic"
pan_processor = AutoImageProcessor.from_pretrained(pan_model_name)
pan_model = Mask2FormerForUniversalSegmentation.from_pretrained(pan_model_name).to(device)
pan_model.eval()

print("Đang tải YOLO11x-seg Instance Model...")
yolo_model = YOLO("yolo11x-seg.pt")

print("✨ TẤT CẢ SOTA MODELS ĐÃ SẴN SÀNG!")
""")

# --- CELL 4: Helpers for Formatting & Packaging ---
add_code("""# BƯỚC 3: Tiện ích xuất dữ liệu chuẩn CVAT (Segmentation mask 1.1 & COCO 1.0)

CITYSCAPES_MAP = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "building",   # wall -> building
    4: "building",   # fence -> building
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "vegetation", # terrain -> vegetation
    10: "sky",
    11: "person",
    12: "person",    # rider -> person
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle"
}

def predict_sota_semantic(img_pil, target_classes, color_map):
    \"\"\"Run Mask2Former Swin-Large for razor-sharp semantic prediction.\"\"\"
    inputs = sem_processor(images=img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = sem_model(**inputs)
    
    pred_map = sem_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[img_pil.size[::-1]]
    )[0].cpu().numpy()

    h, w = pred_map.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for city_id, lab_name in CITYSCAPES_MAP.items():
        if lab_name in target_classes and lab_name in color_map:
            match_pixels = (pred_map == city_id)
            rgb_mask[match_pixels] = color_map[lab_name]

    # Fill unassigned pixels with valid target class (NO background 0,0,0)
    unassigned = np.all(rgb_mask == 0, axis=-1)
    if np.any(unassigned):
        y_coords, _ = np.mgrid[0:h, 0:w]
        top_fallback = "sky" if "sky" in target_classes else ("building" if "building" in target_classes else target_classes[0])
        bottom_fallback = "road" if "road" in target_classes else ("sidewalk" if "sidewalk" in target_classes else target_classes[-1])
        
        rgb_mask[unassigned & (y_coords < int(h * 0.55))] = color_map.get(top_fallback, [128, 64, 128])
        rgb_mask[unassigned & (y_coords >= int(h * 0.55))] = color_map.get(bottom_fallback, [128, 64, 128])

    return rgb_mask, pred_map


def mask_to_polygons(binary_mask):
    \"\"\"Convert binary mask to clean COCO polygon format [[x1, y1, x2, y2, ...]].\"\"\"
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    polygons = []
    for contour in contours:
        contour = contour.squeeze()
        if contour.ndim == 2 and len(contour) >= 3:
            poly = contour.flatten().tolist()
            if len(poly) >= 6 and len(poly) % 2 == 0:
                polygons.append(poly)
    return polygons


def create_semantic_zip(task_dir, out_zip_path, target_classes, color_map):
    \"\"\"Package semantic segmentation task as CVAT 'Segmentation mask 1.1'.\"\"\"
    images = sorted((task_dir / "images").glob("*.jpg"))
    
    # Build labelmap.txt strictly containing valid classes
    lines = ["# label:color_rgb:parts:actions"]
    for cls in target_classes:
        rgb_str = ",".join(map(str, color_map.get(cls, [0, 0, 0])))
        lines.append(f"{cls}:{rgb_str}::")
    labelmap_text = "\\n".join(lines) + "\\n"
    stems_text = "\\n".join(img_p.stem for img_p in images) + "\\n"

    with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("labelmap.txt", labelmap_text)
        zf.writestr("ImageSets/Segmentation/default.txt", stems_text)
        for img_p in images:
            img = Image.open(img_p).convert("RGB")
            mask_rgb, _ = predict_sota_semantic(img, target_classes, color_map)
            buf = io.BytesIO()
            Image.fromarray(mask_rgb).save(buf, format="PNG")
            zf.writestr(f"SegmentationClass/{img_p.stem}.png", buf.getvalue())
    print(f"✅ Đã tạo {out_zip_path.name} ({len(images)} ảnh)")


def create_instance_coco_zip(task_dir, out_zip_path, allowed_classes):
    \"\"\"Package instance segmentation task as CVAT 'COCO 1.0'.\"\"\"
    images = sorted((task_dir / "images").glob("*.jpg"))
    
    categories = [{"id": i + 1, "name": name, "supercategory": ""} for i, name in enumerate(sorted(allowed_classes))]
    cat_name_to_id = {c["name"]: c["id"] for c in categories}

    coco_images = []
    annotations = []
    ann_id = 1

    for img_idx, img_p in enumerate(images, start=1):
        img_pil = Image.open(img_p).convert("RGB")
        w, h = img_pil.size
        coco_images.append({
            "id": img_idx,
            "file_name": img_p.name,
            "width": w,
            "height": h
        })

        results = yolo_model(img_pil, conf=0.20, iou=0.5, verbose=False)[0]
        if results.masks is not None:
            for mask_data, cls_id in zip(results.masks.data, results.boxes.cls):
                cls_name = yolo_model.names[int(cls_id)]
                if cls_name not in cat_name_to_id:
                    continue
                bin_mask = cv2.resize(mask_data.cpu().numpy().astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                polys = mask_to_polygons(bin_mask)
                if not polys:
                    continue
                
                y_indices, x_indices = np.where(bin_mask > 0)
                if len(x_indices) == 0:
                    continue
                xmin, xmax = int(x_indices.min()), int(x_indices.max())
                ymin, ymax = int(y_indices.min()), int(y_indices.max())
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                area = float(bin_mask.sum())

                annotations.append({
                    "id": ann_id,
                    "image_id": img_idx,
                    "category_id": cat_name_to_id[cls_name],
                    "segmentation": polys,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                })
                ann_id += 1

    payload = {
        "images": coco_images,
        "categories": categories,
        "annotations": annotations
    }

    with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("annotations/instances_default.json", json.dumps(payload, indent=2))
    print(f"✅ Đã tạo {out_zip_path.name} ({len(images)} ảnh, {len(annotations)} objects)")
""")

# --- CELL 5: Easy Semantic ---
add_code("""# TÁC VỤ 1: Easy Semantic (20 điểm) — Mask2Former Swin-Large
task_name = "easy_semantic"
task_dir = DATA_DIR / "tiers" / task_name
classes_meta = json.loads((task_dir / "classes.json").read_text())
target_classes = classes_meta["classes"]
color_map = classes_meta["colors"]

out_zip = SUBMISSION_DIR / f"{task_name}.zip"
create_semantic_zip(task_dir, out_zip, target_classes, color_map)
""")

# --- CELL 6: Medium Instance ---
add_code("""# TÁC VỤ 2: Medium Instance (32 điểm) — YOLO11x-seg
task_name = "medium_instance"
task_dir = DATA_DIR / "tiers" / task_name
classes_meta = json.loads((task_dir / "classes.json").read_text())
allowed_classes = classes_meta["classes"]

out_zip = SUBMISSION_DIR / f"{task_name}.zip"
create_instance_coco_zip(task_dir, out_zip, allowed_classes)
""")

# --- CELL 7: Hard Panoptic ---
add_code("""# TÁC VỤ 3: Hard Panoptic (30 điểm) — Mask2Former Panoptic + YOLO11x
task_name = "hard_panoptic"
task_dir = DATA_DIR / "tiers" / task_name
classes_meta = json.loads((task_dir / "classes.json").read_text())
classes = classes_meta["classes"]
stuff_classes = classes_meta["stuff"]
thing_classes = [c for c in classes if c not in stuff_classes]

images = sorted((task_dir / "images").glob("*.jpg"))
categories = [{"id": i + 1, "name": name, "supercategory": ""} for i, name in enumerate(sorted(classes))]
cat_name_to_id = {c["name"]: c["id"] for c in categories}

coco_images = []
annotations = []
ann_id = 1

for img_idx, img_p in enumerate(images, start=1):
    img_pil = Image.open(img_p).convert("RGB")
    w, h = img_pil.size
    coco_images.append({
        "id": img_idx,
        "file_name": img_p.name,
        "width": w,
        "height": h
    })

    # 1. Mask2Former Semantic Stuff Regions
    _, pred_map = predict_sota_semantic(img_pil, classes, classes_meta["colors"])
    for city_id, lab_name in CITYSCAPES_MAP.items():
        if lab_name in stuff_classes and lab_name in cat_name_to_id:
            stuff_mask = (pred_map == city_id).astype(np.uint8)
            if stuff_mask.sum() > 200:
                polys = mask_to_polygons(stuff_mask)
                if polys:
                    y_indices, x_indices = np.where(stuff_mask > 0)
                    bbox = [int(x_indices.min()), int(y_indices.min()),
                            int(x_indices.max() - x_indices.min()), int(y_indices.max() - y_indices.min())]
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_idx,
                        "category_id": cat_name_to_id[lab_name],
                        "segmentation": polys,
                        "area": float(stuff_mask.sum()),
                        "bbox": bbox,
                        "iscrowd": 0
                    })
                    ann_id += 1

    # 2. YOLO11x Instance Thing Objects
    results = yolo_model(img_pil, conf=0.20, verbose=False)[0]
    if results.masks is not None:
        for mask_data, cls_id in zip(results.masks.data, results.boxes.cls):
            cls_name = yolo_model.names[int(cls_id)]
            if cls_name in thing_classes and cls_name in cat_name_to_id:
                bin_mask = cv2.resize(mask_data.cpu().numpy().astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                polys = mask_to_polygons(bin_mask)
                if polys:
                    y_indices, x_indices = np.where(bin_mask > 0)
                    if len(x_indices) > 0:
                        bbox = [int(x_indices.min()), int(y_indices.min()),
                                int(x_indices.max() - x_indices.min()), int(y_indices.max() - y_indices.min())]
                        annotations.append({
                            "id": ann_id,
                            "image_id": img_idx,
                            "category_id": cat_name_to_id[cls_name],
                            "segmentation": polys,
                            "area": float(bin_mask.sum()),
                            "bbox": bbox,
                            "iscrowd": 0
                        })
                        ann_id += 1

payload = {
    "images": coco_images,
    "categories": categories,
    "annotations": annotations
}

out_zip = SUBMISSION_DIR / f"{task_name}.zip"
with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("annotations/instances_default.json", json.dumps(payload, indent=2))
print(f"✅ Đã tạo {out_zip.name} ({len(images)} ảnh, {len(annotations)} regions/objects)")
""")

# --- CELL 8: 6 Checkpoints ---
add_code("""# TÁC VỤ 4: 6 Checkpoints (18 điểm)
cps = [
    ("cp1_holes", "instance"),
    ("cp2_slice", "instance"),
    ("cp3_thin", "semantic"),
    ("cp4_curb", "semantic"),
    ("cp5_occlusion", "instance"),
    ("cp6_coverage", "semantic"),
]

for cp_name, cp_type in cps:
    cp_dir = DATA_DIR / "checkpoints" / cp_name
    classes_meta = json.loads((cp_dir / "classes.json").read_text())
    out_zip = SUBMISSION_DIR / f"{cp_name}.zip"
    
    if cp_type == "semantic":
        create_semantic_zip(cp_dir, out_zip, classes_meta["classes"], classes_meta["colors"])
    else:
        create_instance_coco_zip(cp_dir, out_zip, classes_meta["classes"])

print("✅ Đã hoàn thành 6 checkpoints!")
""")

# --- CELL 9: Inspect Submissions ---
add_code("""# BƯỚC 4: Tự kiểm toàn bộ 9 ZIP đã tạo
!python scripts/inspect_submissions.py --dir submissions
""")

# --- CELL 10: Download All Submissions ---
add_code("""# BƯỚC 5: Đóng gói và tải toàn bộ ZIP về máy
import zipfile
from pathlib import Path

master_zip_path = "day5_all_submissions.zip"
with zipfile.ZipFile(master_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as master_zf:
    for zf_path in sorted(SUBMISSION_DIR.glob("*.zip")):
        master_zf.write(zf_path, arcname=zf_path.name)
        print(f"Đã đóng gói: {zf_path.name}")

print(f"🎉 TẤT CẢ 9 TASKS ĐÃ SẴN SÀNG: {master_zip_path}")

try:
    from google.colab import files
    files.download(master_zip_path)
    print("Đang bắt đầu tải file về máy...")
except Exception as e:
    print("Nếu không chạy trên Colab, các file ZIP đã nằm sẵn trong thư mục submissions/")
""")

notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python"
        },
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

NB_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Successfully generated SOTA notebook: {NB_PATH}")
