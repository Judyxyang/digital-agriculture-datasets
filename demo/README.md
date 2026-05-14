# BananaAI — Demo

Two ways to run the live demo for funder presentations.

---

## Option A — Gradio web app (shareable link)

```bash
pip install "banana-ripeness[demo]"   # installs gradio
python demo/app.py
```

Gradio prints a public `https://….gradio.live` URL you can share with anyone.
No install required on the viewer's end — works in any browser.

Requires a trained checkpoint at `checkpoints/best.pth`.
Without one, the app runs with ImageNet pretrained weights (for demo UI only).

---

## Option B — Standalone HTML (fully offline)

1. Export the ONNX model:
   ```python
   from banana_ripeness.model_exporter import ModelExporter
   ModelExporter().export("checkpoints/best.pth", format="onnx", output_dir="demo")
   ```
   This places `model_int8.onnx` next to `index.html`.

2. Serve the demo folder (needed for WebAssembly):
   ```bash
   python -m http.server 8080 --directory demo
   ```

3. Open `http://localhost:8080` in any browser.

The HTML file loads `model_int8.onnx` via **ONNX Runtime Web** (WebAssembly).
Inference runs entirely in the browser — no data leaves the device.
Works offline once the page is loaded.

> **Note:** Opening `index.html` directly via `file://` is blocked by browsers
> for WebAssembly. Use the `python -m http.server` step above.
