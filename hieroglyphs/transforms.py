"""Image transforms: the single source of truth for both training and inference."""
from torchvision import transforms

# Gentle, domain-aware augmentation (no flips: sign orientation can be meaningful).
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ColorJitter(brightness=0.2),
    transforms.RandomAffine(5, translate=(0.03, 0.03), scale=(0.95, 1.05)),
    transforms.RandomAutocontrast(p=0.15),
    transforms.ToTensor(),
])

# Deterministic. Every dataset image is already 50x75, so no resize here.
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])
