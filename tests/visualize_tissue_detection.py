from src.wsi.reader import WSIReader
from src.wsi.patch_extractor import PatchExtractor
from src.wsi.tissue_detector import TissueDetector

import matplotlib.pyplot as plt
import matplotlib.patches as patches


WSI_PATH = "data/raw/CMU-1-Small-Region.svs"


def main():

    with WSIReader(WSI_PATH) as reader:

        extractor = PatchExtractor(
            reader=reader,
            patch_size=256,
            stride=256,
            level=0
        )

        detector = TissueDetector(
            saturation_threshold=30,
            tissue_percentage_threshold=0.10
        )

        coordinates = extractor.get_patch_coordinates()

        thumbnail = reader.get_thumbnail(width=800)

        slide_width, slide_height = reader.get_dimensions()

        scale_x = thumbnail.width / slide_width
        scale_y = thumbnail.height / slide_height

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.imshow(thumbnail)

        for x, y in coordinates:

            patch = extractor.extract_patch(x, y)

            is_tissue = detector.is_tissue_patch(patch)

            if is_tissue:
                edge_color = "green"
            else:
                edge_color = "red"

            rectangle = patches.Rectangle(
                (x * scale_x, y * scale_y),
                256 * scale_x,
                256 * scale_y,
                fill=False,
                edgecolor=edge_color,
                linewidth=2
            )

            ax.add_patch(rectangle)

        ax.set_title(
            "Tissue Detection: Green = Tissue, Red = Background"
        )

        ax.axis("off")

        plt.show()


if __name__ == "__main__":
    main()