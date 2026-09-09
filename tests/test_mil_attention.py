import torch

from src.models.mil_attention import AttentionMIL


def main():

    # Simulate one WSI containing 20 tissue patches
    num_patches = 20
    feature_dim = 2048

    features = torch.randn(
        num_patches,
        feature_dim
    )

    print("Input feature shape:")
    print(features.shape)

    # Create MIL model
    model = AttentionMIL(
        feature_dim=2048,
        hidden_dim=256,
        num_classes=2
    )

    # Forward pass
    logits, attention_weights = model(
        features
    )

    print("\nLogits shape:")
    print(logits.shape)

    print("\nAttention shape:")
    print(attention_weights.shape)

    print("\nAttention weights:")
    print(attention_weights.squeeze())

    print("\nAttention sum:")
    print(attention_weights.sum().item())


if __name__ == "__main__":
    main()