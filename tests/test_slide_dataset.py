from pathlib import Path

from src.datasets.slide_dataset import SlideDataset


FEATURE_PATH = Path(
    "data/features/CMU-1-Small-Region_features.pt"
)

COORDINATE_PATH = Path(
    "data/features/CMU-1-Small-Region_coordinates.pt"
)


def main():

    slides = [
        {
            "slide_id": "CMU-1-Small-Region",

            "feature_path": FEATURE_PATH,

            "coordinate_path": COORDINATE_PATH,

            # Temporary value for testing only
            "label": 0
        }
    ]

    dataset = SlideDataset(slides)

    print("Number of slides:")
    print(len(dataset))

    slide = dataset[0]

    print("\nSlide ID:")
    print(slide["slide_id"])

    print("\nFeature shape:")
    print(slide["features"].shape)

    print("\nCoordinate shape:")
    print(slide["coordinates"].shape)

    print("\nLabel:")
    print(slide["label"])


if __name__ == "__main__":
    main()