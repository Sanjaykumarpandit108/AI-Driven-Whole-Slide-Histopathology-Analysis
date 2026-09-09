from pathlib import Path
import csv


class PatchDataset:
    """
    Manage extracted WSI patches and their metadata.
    """

    def __init__(self, output_dir="data/processed"):
        self.output_dir = Path(output_dir)

        self.patch_dir = self.output_dir / "patches"
        self.manifest_path = self.output_dir / "patch_manifest.csv"

        self.patch_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def initialize_manifest(self):
        """
        Create the patch metadata CSV file.
        """

        if self.manifest_path.exists():
            return

        with open(
            self.manifest_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "slide_id",
                "patch_id",
                "x",
                "y",
                "patch_size",
                "tissue_percentage",
                "patch_path",
                "label"
            ])

    def save_patch(
        self,
        slide_id,
        patch_id,
        patch,
        x,
        y,
        tissue_percentage,
        label=None
    ):
        """
        Save a patch image and its metadata.
        """

        slide_directory = self.patch_dir / slide_id

        slide_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        filename = f"patch_{patch_id:05d}_x{x}_y{y}.png"

        patch_path = slide_directory / filename

        patch.save(
            patch_path,
            format="PNG"
        )

        with open(
            self.manifest_path,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                slide_id,
                patch_id,
                x,
                y,
                patch.size[0],
                tissue_percentage,
                str(patch_path),
                label
            ])

        return patch_path