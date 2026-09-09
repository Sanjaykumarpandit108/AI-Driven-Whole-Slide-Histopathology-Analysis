from pathlib import Path

from PIL import Image

from src.models.feature_extractor import CNNFeatureExtractor


PATCH_DIR = Path(
    "data/processed/patches/CMU-1-Small-Region"
)


def main():

    patch_files = list(PATCH_DIR.glob("*.png"))

    if not patch_files:
        raise RuntimeError(
            "No extracted patches found."
        )

    first_patch = patch_files[0]

    print("Testing patch:")
    print(first_patch)

    image = Image.open(first_patch)

    print("\nInput image:")
    print("Size:", image.size)
    print("Mode:", image.mode)

    extractor = CNNFeatureExtractor()

    print("\nDevice:")
    print(extractor.device)

    features = extractor.extract(image)

    print("\nFeature vector:")
    print("Shape:", features.shape)
    print("Number of features:", features.numel())

    print("\nFirst 10 values:")

    print(features[:10])


if __name__ == "__main__":
    main()