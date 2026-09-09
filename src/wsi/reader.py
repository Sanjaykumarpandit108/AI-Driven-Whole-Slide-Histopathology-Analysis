from pathlib import Path
from turtle import width

import openslide
from PIL import Image


class WSIReader:
    """
    Wrapper around OpenSlide for reading Whole Slide Images (WSIs).
    """

    def __init__(self, wsi_path):
        self.wsi_path = Path(wsi_path)

        if not self.wsi_path.exists():
            raise FileNotFoundError(
                f"WSI file not found: {self.wsi_path}"
            )

        self.slide = openslide.OpenSlide(str(self.wsi_path))

    def get_dimensions(self):
        """Return the full-resolution WSI dimensions."""
        return self.slide.dimensions

    def get_level_count(self):
        """Return the number of resolution levels."""
        return self.slide.level_count

    def get_level_dimensions(self):
        """Return dimensions of every WSI level."""
        return self.slide.level_dimensions

    def get_level_downsamples(self):
        """Return downsampling factor for every WSI level."""
        return self.slide.level_downsamples

    def get_properties(self):
        """Return WSI metadata/properties."""
        return dict(self.slide.properties)

    def read_region(self, x, y, level, width, height):
        """
        Read a rectangular region from the WSI.

        Parameters
        ----------
        x, y : int
            Top-left coordinate at level 0.

        level : int
            Pyramid level from which to read.

        width, height : int
            Size of the returned region.
        """

        region = self.slide.read_region(
            (x, y),
            level,
            (width, height)
        )

        # OpenSlide returns RGBA.
        # Convert to RGB for normal image processing.
        return region.convert("RGB")

    def close(self):
        """Close the WSI."""
        self.slide.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def get_thumbnail(self, width=800):
        """
        Generate a thumbnail of the WSI while preserving aspect ratio.
        """

        slide_width, slide_height = self.get_dimensions()

        scale = width / slide_width
        height = int(slide_height * scale)

        return self.slide.get_thumbnail((width, height))