from pathlib import Path
import csv

import torch
from PIL import Image

from src.models.feature_extractor import CNNFeatureExtractor


MANIFEST_PATH = Path(
    "data/processed/patch_manifest.csv"
)

FEATURE_DIR = Path(
    "data/features"
)


def main():

    FEATURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load the CNN
    extractor = CNNFeatureExtractor()

    print("Device:", extractor.device)

    # Read manifest
    with open(
        MANIFEST_PATH,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    if not rows:
        raise RuntimeError(
            "No patches found in manifest."
        )

    print(
        "\nNumber of tissue patches:",
        len(rows)
    )

    # Store all features for this slide
    slide_features = []

    coordinates = []

    slide_id = rows[0]["slide_id"]

    for index, row in enumerate(rows):

        patch_path = Path(
            row["patch_path"]
        )

        image = Image.open(
            patch_path
        ).convert("RGB")

        # Extract 2048-D feature vector
        features = extractor.extract(
            image
        )

        slide_features.append(
            features
        )

        coordinates.append([
            int(row["x"]),
            int(row["y"])
        ])

        print(
            f"Processed "
            f"{index + 1}/{len(rows)}: "
            f"{patch_path.name}"
        )

    # Convert list into tensor
    features_tensor = torch.stack(
        slide_features
    )

    coordinates_tensor = torch.tensor(
        coordinates,
        dtype=torch.int32
    )

    # Save
    feature_path = (
        FEATURE_DIR /
        f"{slide_id}_features.pt"
    )

    coordinate_path = (
        FEATURE_DIR /
        f"{slide_id}_coordinates.pt"
    )

    torch.save(
        features_tensor,
        feature_path
    )

    torch.save(
        coordinates_tensor,
        coordinate_path
    )

    print("\n==============================")
    print("Feature extraction completed")
    print("==============================")

    print(
        "Feature tensor:",
        features_tensor.shape
    )

    print(
        "Coordinate tensor:",
        coordinates_tensor.shape
    )

    print(
        "Features saved:",
        feature_path
    )

    print(
        "Coordinates saved:",
        coordinate_path
    )


if __name__ == "__main__":
    main()