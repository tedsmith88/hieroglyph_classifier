"""Single-image inference used by the app."""
import torch
from PIL import Image
from torchvision import transforms

from .config import IMG_HW
from .transforms import test_transform

_resize = transforms.Resize(IMG_HW)  # torchvision reads sizes as (height, width)


def predict_topk(model: torch.nn.Module, labels: list[str], image: Image.Image,
                 k: int = 5) -> list[tuple[str, float]]:
    """Top-k (Gardiner code, probability) for one image.

    Images already in the training size (50x75) are used as they are, exactly like in
    evaluation. Anything else is resized to that size first.
    """
    image = image.convert("RGB")
    if image.size != IMG_HW[::-1]:  # PIL sizes are (width, height)
        image = _resize(image)
    with torch.no_grad():
        probs = torch.softmax(model.eval()(test_transform(image).unsqueeze(0)), dim=1)[0]
    top = torch.topk(probs, k)
    return [(labels[i], p) for p, i in zip(top.values.tolist(), top.indices.tolist())]
