import openslide
import numpy as np
from pathlib import Path


class TissueDetector:

    def __init__(self, slide_path, level=8):
        self.slide_path = Path(slide_path)
        self.level = level

        self.slide = openslide.OpenSlide(str(self.slide_path))

        self.level_dimensions = self.slide.level_dimensions[level]
        self.downsample = self.slide.level_downsamples[level]

    def get_thumbnail(self):
        """
        Read the complete WSI at a low-resolution pyramid level.
        """

        width, height = self.level_dimensions

        thumbnail = self.slide.read_region(
            (0, 0),
            self.level,
            (width, height)
        ).convert("RGB")

        return thumbnail

    def create_tissue_mask(self, threshold=220):
        """
        Create a simple tissue mask.

        Darker pixels are considered tissue.
        """

        thumbnail = self.get_thumbnail()

        image = np.array(thumbnail)

        gray = np.mean(image, axis=2)

        tissue_mask = gray < threshold

        return thumbnail, tissue_mask

    def get_tissue_percentage(self, tissue_mask):
        """
        Calculate the percentage of the thumbnail
        occupied by tissue.
        """

        tissue_pixels = np.sum(tissue_mask)
        total_pixels = tissue_mask.size

        return (tissue_pixels / total_pixels) * 100

    def get_tissue_coordinates(
        self,
        patch_size=256,
        tissue_threshold=0.20
    ):
        """
        Generate level-0 patch coordinates that contain
        enough tissue.
        """

        _, tissue_mask = self.create_tissue_mask()

        mask_height, mask_width = tissue_mask.shape

        coordinates = []

        # Convert patch size from level-0 pixels
        # to the tissue-mask resolution.
        mask_patch_size = max(
            1,
            int(patch_size / self.downsample)
        )

        # Number of level-0 pixels represented
        # by each mask pixel.
        ds = self.downsample

        for mask_y in range(
            0,
            mask_height,
            mask_patch_size
        ):

            for mask_x in range(
                0,
                mask_width,
                mask_patch_size
            ):

                mask_region = tissue_mask[
                    mask_y:min(mask_y + mask_patch_size, mask_height),
                    mask_x:min(mask_x + mask_patch_size, mask_width)
                ]

                if mask_region.size == 0:
                    continue

                tissue_ratio = np.mean(mask_region)

                if tissue_ratio >= tissue_threshold:

                    x = int(mask_x * ds)
                    y = int(mask_y * ds)

                    if x + patch_size <= self.slide.dimensions[0] and \
                    y + patch_size <= self.slide.dimensions[1]:

                        coordinates.append((x, y))

        return coordinates
    def close(self):
        self.slide.close()