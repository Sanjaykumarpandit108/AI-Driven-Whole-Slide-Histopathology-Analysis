from pathlib import Path

from src.wsi.reader import WSIReader
from src.wsi.patch_extractor import PatchExtractor
from src.wsi.tissue_detector import TissueDetector
from src.dataset.patch_dataset import PatchDataset


WSI_PATH = "data/raw/CMU-1-Small-Region.svs"

SLIDE_ID = Path(WSI_PATH).stem


def main():

    print("Slide ID:", SLIDE_ID)

    with WSIReader(WSI_PATH) as reader:

        detector = TissueDetector(
            saturation_threshold=30,
            tissue_percentage_threshold=0.10
        )

        extractor = PatchExtractor(
            reader=reader,
            patch_size=256,
            stride=256,
            level=0,
            tissue_detector=detector
        )

        dataset = PatchDataset(
            output_dir="data/processed"
        )

        dataset.initialize_manifest()

        patch_id = 0
        tissue_count = 0

        print("\nExtracting tissue patches...\n")

        for coordinate, patch in extractor.iter_patches():

            x, y = coordinate

            tissue_percentage = (
                detector.calculate_tissue_percentage(
                    patch
                )
            )

            if tissue_percentage < 0.10:
                continue

            patch_id += 1
            tissue_count += 1

            patch_path = dataset.save_patch(
                slide_id=SLIDE_ID,
                patch_id=patch_id,
                patch=patch,
                x=x,
                y=y,
                tissue_percentage=tissue_percentage
            )

            print(
                f"Saved patch {patch_id}: "
                f"({x}, {y}) "
                f"tissue={tissue_percentage:.2%}"
            )

        print("\n==============================")
        print("Dataset generation completed")
        print("==============================")

        print("Slide:", SLIDE_ID)
        print("Tissue patches:", tissue_count)
        print(
            "Manifest:",
            dataset.manifest_path
        )


if __name__ == "__main__":
    main()