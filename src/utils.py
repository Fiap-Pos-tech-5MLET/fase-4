import torch


def save_model(model, path="model.pth"):
    """Save model state dictionary to file."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path="model.pth"):
    """Load model state dictionary from file."""
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model
