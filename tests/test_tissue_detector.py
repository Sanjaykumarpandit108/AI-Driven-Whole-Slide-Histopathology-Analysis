from src.wsi.tissue_detector import TissueDetector


slide_path = "data/raw/camelyon16/normal/normal_001.tif"

detector = TissueDetector(slide_path)

print("WSI opened successfully!")

print()
print("Detection level:")
print(detector.level)

print()
print("Low-resolution dimensions:")
print(detector.level_dimensions)

thumbnail = detector.get_thumbnail()

print()
print("Thumbnail size:")
print(thumbnail.size)

thumbnail, mask = detector.create_tissue_mask()

tissue_percentage = detector.get_tissue_percentage(mask)

print()
print("Tissue percentage:")
print(f"{tissue_percentage:.2f}%")

detector.close()