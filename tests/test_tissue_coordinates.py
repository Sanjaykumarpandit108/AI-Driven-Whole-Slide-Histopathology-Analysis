from src.wsi.tissue_detector import TissueDetector


slide_path = "data/raw/camelyon16/normal/normal_001.tif"

detector = TissueDetector(slide_path)

print("Finding tissue coordinates...")
print()

coordinates = detector.get_tissue_coordinates(
    patch_size=256,
    tissue_threshold=0.20
)

print("Number of tissue patches:")
print(len(coordinates))

print()
print("First 20 coordinates:")

for coordinate in coordinates[:20]:
    print(coordinate)

detector.close()