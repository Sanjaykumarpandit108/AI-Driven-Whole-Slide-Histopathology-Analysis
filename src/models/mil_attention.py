import torch
import torch.nn as nn


class AttentionMIL(nn.Module):
    """
    Attention-based Multiple Instance Learning model.

    Input:
        N patches × feature_dim

    Output:
        Slide-level prediction
        + attention weights
    """

    def __init__(
        self,
        feature_dim=2048,
        hidden_dim=256,
        num_classes=2
    ):
        super().__init__()

        # Transform each patch feature
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU()
        )

        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Slide-level classifier
        self.classifier = nn.Linear(
            hidden_dim,
            num_classes
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape: [num_patches, feature_dim]

        Returns
        -------
        logits
            Slide-level class scores.

        attention_weights
            Importance assigned to each patch.
        """

        # --------------------------------
        # Step 1: Project patch features
        # --------------------------------

        h = self.feature_projection(x)

        # Shape:
        # [N, 2048]
        #      ↓
        # [N, 256]

        # --------------------------------
        # Step 2: Calculate attention score
        # --------------------------------

        attention_scores = self.attention(h)

        # [N, 1]

        # --------------------------------
        # Step 3: Normalize attention
        # --------------------------------

        attention_weights = torch.softmax(
            attention_scores,
            dim=0
        )

        # --------------------------------
        # Step 4: Weighted aggregation
        # --------------------------------

        slide_representation = torch.sum(
            attention_weights * h,
            dim=0
        )

        # --------------------------------
        # Step 5: Slide classification
        # --------------------------------

        logits = self.classifier(
            slide_representation
        )

        return logits, attention_weights