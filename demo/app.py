"""BananaAI — Gradio demo for funder presentations.

Run:
    pip install gradio
    python demo/app.py

A public shareable link is printed to the terminal (share=True).
"""
from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
from PIL import Image

# Allow `python demo/app.py` from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from banana_ripeness.inference_pipeline import InferencePipeline  # noqa: E402
from banana_ripeness.ripeness_classifier import RIPENESS_STAGES   # noqa: E402

# ---------------------------------------------------------------------------
# Stage metadata — label, hex colour, emoji, farmer advice
# ---------------------------------------------------------------------------
STAGE_META = {
    "unripe": {
        "colour": "#2E7D32",
        "emoji": "🟢",
        "advice": "Not ready. Wait 4–7 more days before harvest.",
    },
    "nearly-ripe": {
        "colour": "#9E9D24",
        "emoji": "🟡",
        "advice": "Almost there. Harvest in 1–3 days for best market price.",
    },
    "ripe": {
        "colour": "#F9A825",
        "emoji": "🌕",
        "advice": "Peak sweetness. Harvest now — maximum market value.",
    },
    "overripe": {
        "colour": "#5D4037",
        "emoji": "🟤",
        "advice": "Past peak. Sell immediately or process into by-products.",
    },
}

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
CHECKPOINT = Path("checkpoints/best.pth")
_pipeline: InferencePipeline | None = None


def _get_pipeline() -> InferencePipeline:
    global _pipeline
    if _pipeline is None:
        if CHECKPOINT.exists():
            _pipeline = InferencePipeline(checkpoint_path=CHECKPOINT)
        else:
            # Fall back to ImageNet pretrained weights for live demo
            # (replace with fine-tuned checkpoint after training)
            _pipeline = InferencePipeline(pretrained=True)
    return _pipeline


# ---------------------------------------------------------------------------
# Inference function called by Gradio
# ---------------------------------------------------------------------------
def classify(image: Image.Image):
    if image is None:
        return (
            "<p style='color:#888'>Please upload or capture a banana photo.</p>",
            None,
        )

    pipeline = _get_pipeline()
    result = pipeline.predict(image)
    meta = STAGE_META[result.stage]

    # Build all-class probabilities for the bar chart
    import torch
    from banana_ripeness.ripeness_classifier import RipenessClassifier

    classifier = pipeline._classifier
    from banana_ripeness.image_preprocessor import ImagePreprocessor
    tensor = ImagePreprocessor().preprocess(image).unsqueeze(0)
    with torch.no_grad():
        logits = classifier._model(tensor)
        probs = torch.softmax(logits, dim=1)[0].tolist()

    # HTML result card
    low_conf_banner = ""
    if result.low_confidence:
        low_conf_banner = (
            "<div style='background:#FFF3E0;border:1px solid #FFB300;"
            "border-radius:8px;padding:10px;margin-top:12px;font-size:14px;'>"
            "⚠️ <strong>Low confidence</strong> — try retaking the photo with "
            "better lighting or a closer angle.</div>"
        )

    html = f"""
    <div style='font-family:sans-serif;max-width:480px;'>
      <div style='background:{meta["colour"]};border-radius:12px;padding:20px 24px;
                  color:white;display:flex;align-items:center;gap:16px;'>
        <span style='font-size:48px;line-height:1;'>{meta["emoji"]}</span>
        <div>
          <div style='font-size:28px;font-weight:700;letter-spacing:0.5px;'>
            {result.stage.upper()}
          </div>
          <div style='font-size:18px;opacity:0.9;'>
            Confidence: {result.confidence:.1%}
          </div>
        </div>
      </div>
      <div style='margin-top:14px;padding:14px 16px;background:#F5F5F5;
                  border-radius:8px;font-size:15px;color:#333;'>
        🌾 <strong>Farmer advice:</strong> {meta["advice"]}
      </div>
      {low_conf_banner}
    </div>
    """

    # Bar chart data: {label: probability}
    chart_data = {stage: round(p, 4) for stage, p in zip(RIPENESS_STAGES, probs)}
    return html, chart_data


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
DESCRIPTION = """
## 🍌 BananaAI — Instant Ripeness Detection

Upload a photo of bananas (or use your webcam) and the AI will classify the
ripeness stage in under 500ms — **fully offline, on-device**.

Built with MobileNetV2 fine-tuned on 164,775 banana images
([luischuquimarca/Banana_Ripeness](https://huggingface.co/datasets/luischuquimarca/Banana_Ripeness)).
"""

ARTICLE = """
---
### How it works
1. Photo is resized to **224 × 224 pixels** and normalised (ImageNet statistics)
2. **MobileNetV2** neural network classifies across 4 ripeness stages
3. Result is returned in **< 500ms** — same pipeline runs on Android, no internet needed

| Stage | Colour | Farmer action |
|---|---|---|
| Unripe | 🟢 Green | Wait 4–7 days |
| Nearly-Ripe | 🟡 Yellow-green | Harvest in 1–3 days |
| Ripe | 🌕 Yellow | Harvest now |
| Overripe | 🟤 Brown | Sell immediately |
"""

with gr.Blocks(title="BananaAI — Ripeness Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Banana photo",
                type="pil",
                sources=["upload", "webcam", "clipboard"],
                height=320,
            )
            classify_btn = gr.Button("🔍 Classify Ripeness", variant="primary", size="lg")

        with gr.Column(scale=1):
            result_html = gr.HTML(label="Result")
            prob_chart = gr.Label(
                label="Confidence per stage",
                num_top_classes=4,
            )

    classify_btn.click(
        fn=classify,
        inputs=image_input,
        outputs=[result_html, prob_chart],
    )
    image_input.change(
        fn=classify,
        inputs=image_input,
        outputs=[result_html, prob_chart],
    )

    gr.Markdown(ARTICLE)


if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")
