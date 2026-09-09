from pathlib import Path

import torch
from torch.utils.data import Dataset


class SlideDataset(Dataset):
    """
    Dataset for slide-level Multiple Instance Learning.

    Each item represents ONE whole slide.

    A slide contains:
        - patch-level feature vectors
        - patch coordinates
        - slide-level label
    """

    def __init__(self, slides):
        """
        Parameters
        ----------
        slides : list of dict

            Example:

            [
                {
                    "slide_id": "slide_001",
                    "feature_path": "...",
                    "coordinate_path": "...",
                    "label": 1
                }
            ]
        """

        self.slides = slides

    def __len__(self):
        """
        Number of slides.
        """

        return len(self.slides)

    def __getitem__(self, index):
        """
        Load one complete slide.
        """

        slide = self.slides[index]

        # Load CNN features
        features = torch.load(
            slide["feature_path"],
            weights_only=True
        )

        # Load patch coordinates
        coordinates = torch.load(
            slide["coordinate_path"],
            weights_only=True
        )

        # Slide-level label
        label = torch.tensor(
            slide["label"],
            dtype=torch.long
        )

        return {
            "slide_id": slide["slide_id"],
            "features": features,
            "coordinates": coordinates,
            "label": label
        }