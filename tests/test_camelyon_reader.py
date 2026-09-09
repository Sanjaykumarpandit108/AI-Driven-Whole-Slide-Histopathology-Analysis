import openslide
from pathlib import Path


slide_path = Path("data/raw/camelyon16/normal/normal_001.tif")

print("Checking:", slide_path)
print()

if not slide_path.exists():
    print("ERROR: WSI file not found!")
    raise SystemExit(1)

print("File exists!")
print(f"File size: {slide_path.stat().st_size / (1024**3):.2f} GB")
print()

try:
    slide = openslide.OpenSlide(str(slide_path))

    print("WSI opened successfully!")
    print()
    
    print("Dimensions:")
    print(slide.dimensions)
    print()

    print("Number of pyramid levels:")
    print(slide.level_count)
    print()

    print("Level dimensions:")
    print(slide.level_dimensions)
    print()

    print("Level downsamples:")
    print(slide.level_downsamples)
    print()

    print("Number of metadata properties:")
    print(len(slide.properties))

    slide.close()

except Exception as e:
    print("ERROR while opening WSI:")
    print(e)