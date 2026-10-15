# Egyptian hieroglyph classifier

Single-sign classification of Egyptian hieroglyphs into **170 Gardiner sign classes**, using a transfer-learned ResNet-18, with a small Streamlit demo.

It is a learning project, so the repository keeps the whole process, including the bugs and the dead ends (see [What I learned](#what-i-learned)).

| | |
|---|---|
| Model | ResNet-18 (ImageNet weights), last block (`layer4`) fine-tuned, new 170-way head |
| Input | one sign, 50 px wide x 75 px tall |
| Train accuracy | 95.1% (3,584 images) |
| **Test accuracy** | **79.3% top-1** (444 held-out images from a different picture of the source book) |

The test number comes from one seed (seed 0). On the dev split, seed-to-seed spread was 1.3 to 5.9 points, so treat 79.3% as roughly "high 70s", not as a precise figure.

## Demo

```bash
./venv/bin/streamlit run app.py
```

- Upload a picture of a single sign, or draw a random sample from the held-out test set.
- Shows the top-5 predictions as confidence bars, and whether a test sample was classified correctly.
- For uploaded images, warns when the top confidence is below 40%.

## Limitations

- **Closed set.** It only knows 170 signs and always names one, even for an unrelated image. The low-confidence warning is a rough threshold, not calibrated probability.
- **Train/test gap.** Accuracy is 95% on training images and 79% on the test images. Test images come from a different picture of the source book, with different scans and lighting. A split carved out of the training data gives about 86%, so a random split would overstate real-world accuracy.
- **Small classes.** 40 of the 170 classes have a single training image.
- **Noisy labels.** The dataset's author states that the labelling may not be 100% correct.
- It classifies **single, already-cropped signs**. It does not find or read signs in a full inscription.

## How it works

- **Data:** one folder per class for training; the test folder is flat, with the label in the file name (`070048_F34.png` is sign `F34`). Test files whose sign is not a training class (`UNKNOWN`, `M20`) are skipped, so label indices can never silently disagree.
- **Model:** ResNet-18 pretrained on ImageNet. The backbone is frozen except `layer4`, and `fc` is replaced by `Linear(512, 170)`.
- **Training:** Adam with two learning rates (head `1e-3`, `layer4` `1e-4`), batch size 32, 10 epochs, seed 0, no weight decay.
- **Augmentation:** gentle and domain-aware (brightness, small affine, occasional autocontrast). No flips, because the orientation of a sign can carry meaning.
- **Evaluation:** configurations are compared on a stratified 20% dev split of the training data. The test set is used once, for the final model.
- **One source of truth:** the transforms live in `hieroglyphs/transforms.py` and are used by training, evaluation and the app.

## What I learned

The full experiment log, runnable top to bottom, is in [`Notebook.ipynb`](Notebook.ipynb).

| Question | Answer |
|---|---|
| Frozen backbone only? | High bias (about 57% test). Unfreeze `layer4`. |
| More capacity (`layer3`)? | Train went up, test stayed flat, an early variance signal. Reverted. |
| Simple vs gentle augmentation? | Within seed noise. |
| Photometric augmentation (contrast/blur)? | Train hit about 99% (memorisation), dev was no better. Ruled out. |
| Weight decay? | No effect. With BatchNorm after every conv, the network is invariant to the scale of the conv weights. |
| Why is test (79%) below dev (86%)? | A real distribution gap between the train and test scans, not a modelling bug. |

Two bugs worth remembering:

1. **Silent evaluation bugs.** `model.eval()` was missing during evaluation, so BatchNorm normalised with per-batch statistics, and "train accuracy" was measured on augmented images. Both are fixed by construction: `evaluate()` always switches to eval mode, and train accuracy uses a clean, non-augmented view.
2. **Preprocessing drift.** The app had its own copy of the preprocessing with `Resize((50, 75))`. `Resize` reads sizes as (height, width), but the images are 50 wide and 75 tall, so the app squashed every tall sign. For `070048_F34.png` the notebook said F34 (78.0%) and the app said I9 (16.4%). The fix was a single definition of the transforms plus tests that guard it (`tests/test_pipeline.py`).

## Setup

Python 3.14 with the versions pinned in `requirements.txt`.

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

**1. Dataset.** The images are not part of this repository. They live in a separate Hugging Face dataset repo. Install [git-lfs](https://git-lfs.com) first, then clone it into the project root:

```bash
git lfs install
git clone https://huggingface.co/datasets/HamdiJr/Egyptian_hieroglyphs
```

The code expects the images at `Egyptian_hieroglyphs/Dataset/train` and `Egyptian_hieroglyphs/Dataset/test`.

**2. Weights.** `weights/labels.csv` (class index to Gardiner code) is tracked. `weights/weights.pt` (about 45 MB) is not, so build it:

```bash
./venv/bin/python train.py --overwrite
```

On the author's machine one epoch takes about 100 seconds. For a quick end-to-end check, run `train.py --epochs 1 --out-dir runs/smoke`.

**3. Run the app and the tests.**

```bash
./venv/bin/streamlit run app.py
./venv/bin/python -m pytest            # fast tests
./venv/bin/python -m pytest -m ""      # also the full test-set accuracy check
```

The tests and the app need both the dataset and `weights/weights.pt`.

## Project layout

```
hieroglyphs/        reusable package
  config.py           paths and hyperparameters
  data.py             datasets, loaders, stratified dev split
  transforms.py       train and test transforms (single source of truth)
  model.py            ResNet-18 builder and weight loading
  training.py         training loop and checkpoint saving
  evaluate.py         top-1 accuracy (always in eval mode)
  predict.py          single-image top-k inference used by the app
  labels.py           labels.csv reading and writing
train.py            final pipeline: train, save weights and labels, report accuracy
app.py              Streamlit demo (UI only)
Notebook.ipynb      experiment log
tests/              guards against preprocessing drift and label mismatches
weights/            labels.csv (tracked); weights.pt is built locally
```

## Data and credits

The images come from the Hugging Face dataset [`HamdiJr/Egyptian_hieroglyphs`](https://huggingface.co/datasets/HamdiJr/Egyptian_hieroglyphs), built from signs found in pictures of the book *The Pyramid of Unas* (A. Piankoff, 1955) and labelled according to the Gardiner Sign List. The dataset's README states its licence as "GPL - non commercial use". Respect that for the images and, to be safe, for models trained on them: this is a non-commercial learning project. This repository does not contain the images.

## Possible next steps

- Log the mean loss per epoch (training currently prints only the last batch's loss).
- Calibrate the confidence scores and replace the fixed 40% warning threshold.
- Show information about a predicted sign (for example from a lexicon) in the app.
