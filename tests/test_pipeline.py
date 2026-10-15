"""Guards the pipeline against the bugs this project already had (drifting preprocessing)."""
import pytest
import torch
from PIL import Image
from torchvision.datasets import ImageFolder

from hieroglyphs.config import NUM_CLASSES, TEST_DIR, TRAIN_DIR, WEIGHTS_DIR
from hieroglyphs.data import HieroglyphTestDataset, make_loader
from hieroglyphs.evaluate import evaluate
from hieroglyphs.labels import load_labels
from hieroglyphs.model import build_model, load_weights
from hieroglyphs.predict import predict_topk
from hieroglyphs.transforms import test_transform as eval_transform

SAMPLE = TEST_DIR / "070048_F34.png"  # the image on which app and notebook once disagreed


@pytest.fixture(scope="module")
def model():
    return load_weights(build_model(pretrained=False), WEIGHTS_DIR / "weights.pt")


@pytest.fixture(scope="module")
def labels():
    return load_labels()


def test_labels_match_train_folder(labels):
    assert len(labels) == NUM_CLASSES
    assert labels == ImageFolder(TRAIN_DIR).classes


def test_known_sample_is_f34(model, labels):
    top1, prob = predict_topk(model, labels, Image.open(SAMPLE), k=3)[0]
    assert top1 == "F34"
    assert prob > 0.5


def test_predict_matches_evaluation_preprocessing(model, labels):
    """Dataset images must go through exactly test_transform, with no extra resize."""
    image = Image.open(SAMPLE)
    with torch.no_grad():
        probs = torch.softmax(model(eval_transform(image).unsqueeze(0)), dim=1)[0]
    expected = probs.max().item()
    assert predict_topk(model, labels, image, k=1)[0][1] == pytest.approx(expected, abs=1e-6)


def test_other_sizes_are_resized(model, labels):
    big = Image.open(SAMPLE).convert("RGB").resize((200, 300))
    assert len(predict_topk(model, labels, big, k=5)) == 5


@pytest.mark.slow
def test_test_set_accuracy(model, labels):
    test_set = HieroglyphTestDataset({c: i for i, c in enumerate(labels)}, transform=eval_transform)
    assert len(test_set) == 444
    assert evaluate(model, make_loader(test_set)) == pytest.approx(0.7928, abs=0.0005)
