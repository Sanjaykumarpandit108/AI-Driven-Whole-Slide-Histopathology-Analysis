from pathlib import Path

import torch

from src.models.mil_attention import AttentionMIL


FEATURE_PATH = Path(
    "data/features/CMU-1-Small-Region_features.pt"
)


def main():

    # Load real CNN features
    features = torch.load(
        FEATURE_PATH,
        weights_only=True
    )

    print("Loaded features:")
    print(features.shape)

    # Create MIL model
    model = AttentionMIL(
        feature_dim=2048,
        hidden_dim=256,
        num_classes=2
    )

    # Run MIL
    logits, attention_weights = model(
        features
    )

    print("\nMIL output:")
    print("Logits:", logits.shape)

    print(
        "Attention:",
        attention_weights.shape
    )

    print(
        "\nAttention sum:",
        attention_weights.sum().item()
    )

    print("\nAttention per patch:")

    for i, weight in enumerate(
        attention_weights.squeeze()
    ):

        print(
            f"Patch {i + 1}: "
            f"{weight.item():.6f}"
        )


if __name__ == "__main__":
    main()