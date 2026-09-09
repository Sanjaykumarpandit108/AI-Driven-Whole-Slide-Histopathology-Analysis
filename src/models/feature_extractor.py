import torch
import torch.nn as nn

from torchvision.models import (
    resnet50,
    ResNet50_Weights
)


class CNNFeatureExtractor:
    """
    Pretrained ResNet-50 feature extractor.

    Converts pathology image patches into
    fixed-dimensional feature embeddings.
    """

    def __init__(self, device=None):

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        # Load pretrained ResNet-50
        weights = ResNet50_Weights.DEFAULT

        model = resnet50(weights=weights)

        # Remove final classification layer
        self.model = nn.Sequential(
            *list(model.children())[:-1]
        )

        self.model = self.model.to(self.device)

        # Evaluation mode
        self.model.eval()

        # Image preprocessing associated
        # with the pretrained weights
        self.transform = weights.transforms()

    @torch.no_grad()
    def extract(self, image):
        """
        Extract a feature vector from one image.

        Parameters
        ----------
        image : PIL.Image.Image

        Returns
        -------
        torch.Tensor
            Feature vector with shape [2048].
        """

        image = image.convert("RGB")

        tensor = self.transform(image)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device)

        features = self.model(tensor)

        # [1, 2048, 1, 1]
        #        ↓
        # [2048]
        features = features.squeeze()

        return features.cpu()