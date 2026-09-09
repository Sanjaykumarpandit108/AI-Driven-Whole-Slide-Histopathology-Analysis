from src.wsi.reader import WSIReader
import matplotlib.pyplot as plt


WSI_PATH = "data/raw/CMU-1-Small-Region.svs"


def main():

    with WSIReader(WSI_PATH) as reader:

        # --------------------------------
        # WSI INFORMATION
        # --------------------------------

        print("WSI opened successfully!")

        print("\nDimensions:")
        print(reader.get_dimensions())

        print("\nNumber of levels:")
        print(reader.get_level_count())

        print("\nLevel dimensions:")
        print(reader.get_level_dimensions())

        print("\nLevel downsamples:")
        print(reader.get_level_downsamples())

        print("\nNumber of metadata properties:")
        print(len(reader.get_properties()))

        # --------------------------------
        # READ A PATCH
        # --------------------------------

        patch = reader.read_region(
            x=1000,
            y=1000,
            level=0,
            width=256,
            height=256
        )

        print("\nPatch type:")
        print(type(patch))

        print("\nPatch size:")
        print(patch.size)

        # --------------------------------
        # DISPLAY PATCH
        # --------------------------------

        plt.figure(figsize=(6, 6))
        plt.imshow(patch)
        plt.axis("off")
        plt.title("256 × 256 WSI Patch")
        plt.show()


if __name__ == "__main__":
    main()