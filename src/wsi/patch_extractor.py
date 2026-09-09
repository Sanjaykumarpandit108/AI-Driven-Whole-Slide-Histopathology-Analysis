from src.wsi.reader import WSIReader
from src.wsi.tissue_detector import TissueDetector


class PatchExtractor:
    """
    Extract fixed-size patches from a Whole Slide Image.
    """

    def __init__(
        self,
        reader: WSIReader,
        patch_size=256,
        stride=256,
        level=0,
        tissue_detector=None
    ):
        self.reader = reader
        self.patch_size = patch_size
        self.stride = stride
        self.level = level
        self.tissue_detector = tissue_detector

    def get_patch_coordinates(self):
        """
        Generate top-left coordinates for patches.
        """

        width, height = self.reader.get_dimensions()

        coordinates = []

        for y in range(
            0,
            height - self.patch_size + 1,
            self.stride
        ):
            for x in range(
                0,
                width - self.patch_size + 1,
                self.stride
            ):
                coordinates.append((x, y))

        return coordinates

    def extract_patch(self, x, y):
        """
        Extract a single patch from the WSI.
        """

        return self.reader.read_region(
            x=x,
            y=y,
            level=self.level,
            width=self.patch_size,
            height=self.patch_size
        )

    def iter_patches(self):
        """
        Lazily generate all patches.

        Yields
        ------
        tuple
            ((x, y), patch)
        """

        coordinates = self.get_patch_coordinates()

        for x, y in coordinates:

            patch = self.extract_patch(x, y)

            yield (x, y), patch

    def iter_tissue_patches(self):
        """
        Lazily generate only patches containing sufficient tissue.

        Yields
        ------
        tuple
            ((x, y), patch)
        """

        if self.tissue_detector is None:
            raise ValueError(
                "TissueDetector must be provided "
                "to use iter_tissue_patches()."
            )

        for coordinate, patch in self.iter_patches():

            if self.tissue_detector.is_tissue_patch(patch):

                yield coordinate, patch


    def analyze_patches(self):
        """
        Analyze patches in a single pass.

        Returns patch statistics without storing
        all patches in memory.
        """

        if self.tissue_detector is None:
            raise ValueError(
                "TissueDetector must be provided "
                "to use analyze_patches()."
            )

        total_patches = 0
        tissue_patches = 0

        for coordinate, patch in self.iter_patches():

            total_patches += 1

            if self.tissue_detector.is_tissue_patch(patch):
                tissue_patches += 1

                yield {
                    "coordinate": coordinate,
                    "patch": patch,
                    "is_tissue": True
                }

        # Note:
        # Statistics are handled separately because
        # this method is a generator.