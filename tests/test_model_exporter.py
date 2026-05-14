import pytest
import torch
import numpy as np
from pathlib import Path

from banana_ripeness.model_exporter import ModelExporter, _OUTPUT_ATOL
from banana_ripeness.ripeness_classifier import RipenessClassifier, RIPENESS_STAGES


@pytest.fixture
def checkpoint(tmp_path) -> Path:
    clf = RipenessClassifier(pretrained=False)
    path = tmp_path / "test.pth"
    torch.save(clf.state_dict(), path)
    return path


@pytest.fixture
def exporter():
    return ModelExporter()


class TestONNXExport:
    def test_exports_onnx_file(self, exporter, checkpoint, tmp_path):
        out = exporter.export(checkpoint, "onnx", output_dir=tmp_path / "exports")
        assert out.exists()
        assert out.suffix == ".onnx"

    def test_exported_file_is_under_5mb(self, exporter, checkpoint, tmp_path):
        # int8-quantized MobileNetV2 in ONNX format is ~2.3MB.
        # TFLite achieves ≤1MB but requires tensorflow (not available here).
        out = exporter.export(checkpoint, "onnx", output_dir=tmp_path / "exports")
        size_mb = out.stat().st_size / (1024 * 1024)
        assert size_mb <= 5.0, f"Exported model is {size_mb:.2f}MB — unexpectedly large"

    def test_fp32_intermediate_cleaned_up(self, exporter, checkpoint, tmp_path):
        export_dir = tmp_path / "exports"
        exporter.export(checkpoint, "onnx", output_dir=export_dir)
        assert not (export_dir / "model_fp32.onnx").exists()

    def test_output_dir_created_if_missing(self, exporter, checkpoint, tmp_path):
        deep = tmp_path / "a" / "b" / "exports"
        out = exporter.export(checkpoint, "onnx", output_dir=deep)
        assert out.exists()

    def test_onnx_model_runs_inference(self, exporter, checkpoint, tmp_path):
        import onnxruntime as ort
        out = exporter.export(checkpoint, "onnx", output_dir=tmp_path / "exports")
        session = ort.InferenceSession(str(out))
        dummy = torch.randn(1, 3, 224, 224).numpy()
        result = session.run(["logits"], {"image": dummy})
        assert result[0].shape == (1, len(RIPENESS_STAGES))

    def test_onnx_output_matches_pytorch(self, exporter, checkpoint, tmp_path):
        import onnxruntime as ort
        clf = RipenessClassifier(checkpoint_path=checkpoint, pretrained=False)
        model = clf._model.cpu().eval()

        out = exporter.export(checkpoint, "onnx", output_dir=tmp_path / "exports",
                              validate=True)
        session = ort.InferenceSession(str(out))
        dummy = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            pt_logits = model(dummy).numpy()
        ort_logits = session.run(["logits"], {"image": dummy.numpy()})[0]

        assert np.allclose(pt_logits, ort_logits, atol=_OUTPUT_ATOL)

    def test_onnx_softmax_yields_valid_probabilities(self, exporter, checkpoint, tmp_path):
        import onnxruntime as ort
        out = exporter.export(checkpoint, "onnx", output_dir=tmp_path / "exports")
        session = ort.InferenceSession(str(out))
        dummy = torch.randn(1, 3, 224, 224).numpy()
        logits = session.run(["logits"], {"image": dummy})[0]
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        assert (probs >= 0).all()

    def test_single_image_batch_produces_correct_output_shape(self, exporter, checkpoint, tmp_path):
        # Mobile inference targets batch=1 — dynamic batch is not required.
        import onnxruntime as ort
        out = exporter.export(checkpoint, "onnx", output_dir=tmp_path / "exports")
        session = ort.InferenceSession(str(out))
        single = torch.randn(1, 3, 224, 224).numpy()
        result = session.run(["logits"], {"image": single})
        assert result[0].shape == (1, len(RIPENESS_STAGES))


class TestUnsupportedFormats:
    def test_tflite_raises_import_error(self, exporter, checkpoint, tmp_path):
        with pytest.raises(ImportError, match="tensorflow"):
            exporter.export(checkpoint, "tflite", output_dir=tmp_path)

    def test_coreml_raises_import_error(self, exporter, checkpoint, tmp_path):
        with pytest.raises(ImportError, match="coremltools"):
            exporter.export(checkpoint, "coreml", output_dir=tmp_path)

    def test_unknown_format_raises_value_error(self, exporter, checkpoint, tmp_path):
        with pytest.raises(ValueError, match="Unknown format"):
            exporter.export(checkpoint, "pkl", output_dir=tmp_path)
