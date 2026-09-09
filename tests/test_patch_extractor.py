from src.wsi.reader import WSIReader
from src.wsi.patch_extractor import PatchExtractor


WSI_PATH = "data/raw/CMU-1-Small-Region.svs"


def main():

    with WSIReader(WSI_PATH) as reader:

        extractor = PatchExtractor(
            reader=reader,
            patch_size=256,
            stride=256,
            level=0
        )

        count = 0

        for coordinate, patch in extractor.iter_patches():

            count += 1

            if count <= 10:
                print(
                    f"Patch {count}: "
                    f"coordinate={coordinate}, "
                    f"size={patch.size}"
                )

        print("\nTotal patches:", count)


if __name__ == "__main__":
    main()