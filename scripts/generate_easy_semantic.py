"""Generate easy_semantic.zip matching CVAT Segmentation mask 1.1 specification."""

import io
import json
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "data" / "tiers" / "easy_semantic"
SUBMISSION_DIR = ROOT / "submissions"
SUBMISSION_DIR.mkdir(exist_ok=True)

# Colors from classes.json
COLORS = {
    "road": (128, 64, 128),
    "sidewalk": (244, 35, 232),
    "building": (70, 70, 70),
    "vegetation": (107, 142, 35),
    "sky": (70, 130, 180),
}

LABELMAP_CONTENT = """# label:color_rgb:parts:actions
road:128,64,128::
sidewalk:244,35,232::
building:70,70,70::
vegetation:107,142,35::
sky:70,130,180::
"""


def segment_image(img_path: Path) -> np.ndarray:
    """Produce realistic semantic mask for easy_semantic image."""
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    h, w, _ = arr.shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Convert to float for analysis
    r = arr[..., 0].astype(float)
    g = arr[..., 1].astype(float)
    b = arr[..., 2].astype(float)

    # 1. Sky detection (top half, bright/blueish)
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    sky_mask = (y_coords < int(h * 0.55)) & (b >= 100) & (b >= r - 15) & (g >= 90) & (b >= g - 25)
    mask[sky_mask] = COLORS["sky"]

    # 2. Vegetation (greenish)
    veg_mask = (g > r + 8) & (g > b + 5) & (y_coords < int(h * 0.8))
    mask[veg_mask] = COLORS["vegetation"]

    # 3. Building (upper-middle non-sky non-veg)
    bldg_mask = (y_coords < int(h * 0.6)) & (~sky_mask) & (~veg_mask)
    mask[bldg_mask] = COLORS["building"]

    # 4. Road & Sidewalk (bottom region)
    ground_mask = y_coords >= int(h * 0.55)
    
    # Sidewalk near edges or where lightness/color indicates concrete
    road_mask = ground_mask & (np.abs(x_coords - w / 2) < (w * 0.40)) & (y_coords >= int(h * 0.62))
    mask[ground_mask] = COLORS["sidewalk"]
    mask[road_mask] = COLORS["road"]

    # Refine bottom-center road
    vanish_y = int(h * 0.52)
    left_bound = (w * 0.5) - ((y_coords - vanish_y) / max(1, (h - vanish_y))) * (w * 0.45)
    right_bound = (w * 0.5) + ((y_coords - vanish_y) / max(1, (h - vanish_y))) * (w * 0.45)
    road_triangle = (y_coords >= vanish_y) & (x_coords >= left_bound) & (x_coords <= right_bound)
    mask[road_triangle] = COLORS["road"]

    # Fill any remaining unassigned with building (top) or sidewalk (bottom)
    unassigned = np.all(mask == 0, axis=-1)
    mask[unassigned & (y_coords < int(h * 0.55))] = COLORS["building"]
    mask[unassigned & (y_coords >= int(h * 0.55))] = COLORS["sidewalk"]

    return mask


def main():
    images = sorted((TASK_DIR / "images").glob("*.jpg"))
    zip_path = SUBMISSION_DIR / "easy_semantic.zip"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("labelmap.txt", LABELMAP_CONTENT)
        # CVAT requires ImageSets/Segmentation/default.txt listing image stems
        stems_txt = "\n".join(img_p.stem for img_p in images) + "\n"
        zf.writestr("ImageSets/Segmentation/default.txt", stems_txt)

        for img_p in images:
            mask_rgb = segment_image(img_p)
            mask_img = Image.fromarray(mask_rgb)
            buf = io.BytesIO()
            mask_img.save(buf, format="PNG")
            zf.writestr(f"SegmentationClass/{img_p.stem}.png", buf.getvalue())
            print(f"Added SegmentationClass/{img_p.stem}.png")

    print(f"Created submission: {zip_path}")


if __name__ == "__main__":
    main()
