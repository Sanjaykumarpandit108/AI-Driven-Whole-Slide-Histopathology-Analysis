import openslide
from pathlib import Path


slide_path = Path(
    "data/raw/camelyon16/normal/normal_001.tif"
)

output_path = Path(
    "data/processed/camelyon_test_patch.png"
)


slide = openslide.OpenSlide(str(slide_path))

print("WSI opened successfully!")

# Read a small 256 × 256 region
x = 10000
y = 10000
size = 256

patch = slide.read_region(
    (x, y),
    0,
    (size, size)
)

# OpenSlide returns RGBA
patch = patch.convert("RGB")

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

patch.save(output_path)

slide.close()

print("Patch extracted successfully!")
print("Coordinates:", (x, y))
print("Patch size:", patch.size)
print("Saved to:", output_path)