from src.wsi.reader import WSIReader
from src.wsi.patch_extractor import PatchExtractor
from src.wsi.tissue_detector import TissueDetector


WSI_PATH = "data/raw/CMU-1-Small-Region.svs"


def main():

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

        total_patches = 0
        tissue_patches = 0

        # Count all candidate patches
        for coordinate, patch in extractor.iter_patches():
            total_patches += 1

        # Count only tissue patches
        for coordinate, patch in extractor.iter_tissue_patches():
            tissue_patches += 1

        background_patches = total_patches - tissue_patches

        print("\n========== PATCH SUMMARY ==========")

        print("Total candidate patches:", total_patches)
        print("Tissue patches:", tissue_patches)
        print("Background patches:", background_patches)

        if total_patches > 0:

            tissue_ratio = (
                tissue_patches / total_patches
            )

            print(
                f"Tissue retention rate: "
                f"{tissue_ratio:.2%}"
            )


if __name__ == "__main__":
    main()