"""Streamlit demo. UI only: all ML logic lives in the hieroglyphs package.

    ./venv/bin/streamlit run app.py
"""
import random

import streamlit as st
from PIL import Image
from streamlit import runtime

if not runtime.exists():  # launched with plain `python app.py`, e.g. the editor's Run button
    raise SystemExit("This is a Streamlit app. Start it with:\n\n    ./venv/bin/streamlit run app.py\n")

from hieroglyphs.config import TEST_DIR, WEIGHTS_DIR
from hieroglyphs.data import HieroglyphTestDataset
from hieroglyphs.labels import load_labels
from hieroglyphs.model import build_model, load_weights
from hieroglyphs.predict import predict_topk

TOP_K = 5
LOW_CONFIDENCE = 0.4


@st.cache_resource
def get_model():
    return load_weights(build_model(pretrained=False), WEIGHTS_DIR / "weights.pt")


@st.cache_data
def get_labels() -> list[str]:
    return load_labels()


@st.cache_data
def get_samples() -> list[str]:
    """Test files whose sign is one of the 170 classes the model knows."""
    known = set(get_labels())
    return sorted(
        p.name for p in TEST_DIR.glob("*.png") if HieroglyphTestDataset.label_of(p.name) in known
    )


def show_prediction(image: Image.Image, truth: str | None = None) -> None:
    """Image on the left; top-1 verdict and the top-k confidence bars on the right."""
    results = predict_topk(get_model(), get_labels(), image, k=TOP_K)
    best, confidence = results[0]

    left, right = st.columns([1, 2])
    left.image(image, caption="Your image", width=150)
    with right:
        st.metric("Top prediction", best, f"{confidence:.1%} confidence", delta_color="off")
        if truth is not None:
            if truth == best:
                st.success(f"Correct. The sample is {truth}.")
            else:
                st.error(f"Wrong. The sample is {truth}.")
        elif confidence < LOW_CONFIDENCE:
            st.warning("Low confidence. This may not be one of the 170 signs it knows.")

    st.subheader(f"Top {TOP_K}")
    for code, prob in results:
        st.progress(prob, text=f"**{code}**  ·  {prob:.1%}")


st.set_page_config(page_title="Hieroglyph classifier", page_icon="🏺")
st.title("🏺 Egyptian hieroglyph classifier")
st.write("Upload a picture of a single sign, or try a sample from the held-out test set.")

with st.sidebar:
    st.header("About")
    st.markdown(
        "- ResNet-18, fine-tuned (last block + new head)\n"
        "- 170 Gardiner sign classes\n"
        "- **79.3% top-1** on 444 held-out test images"
    )
    st.caption(
        "It only knows these 170 signs and always names one, even for an unrelated image. "
        "Test scans differ from training scans (lighting, manuscripts), which is why test "
        "accuracy (79%) is lower than training accuracy (95%)."
    )

upload_tab, sample_tab = st.tabs(["Upload an image", "Test-set sample"])

with upload_tab:
    uploaded = st.file_uploader("A single hieroglyph", type=["png", "jpg", "jpeg"])
    if uploaded:
        show_prediction(Image.open(uploaded))

with sample_tab:
    samples = get_samples()
    if not samples:
        st.info("The test dataset is not available on this machine.")
    else:
        if st.button("🎲 Random sample") or "sample" not in st.session_state:
            st.session_state["sample"] = random.choice(samples)
        name = st.session_state["sample"]
        show_prediction(Image.open(TEST_DIR / name), truth=HieroglyphTestDataset.label_of(name))
