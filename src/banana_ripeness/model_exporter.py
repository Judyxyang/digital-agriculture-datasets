from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from banana_ripeness.ripeness_classifier import RipenessClassifier, RIPENESS_STAGES

Format = Literal["onnx", "tflite", "coreml"]

# Dummy input matching ImagePreprocessor output: batch=1, RGB, 224×224
_DUMMY_INPUT = torch.randn(1, 3, 224, 224)

# Tolerance for output equivalence check between PyTorch and exported model
_OUTPUT_ATOL = 1e-4


class ModelExporter:
    """Converts a trained PyTorch checkpoint to mobile-ready ONNX, TFLite, or Core ML.

    ONNX export with int8 quantization is fully supported and tested.
    TFLite requires `tensorflow`; Core ML requires `coremltools`.

    Interface: exporter.export(checkpoint_path, format, output_dir) → Path
    """

    def export(
        self,
        checkpoint_path: str | Path,
        format: Format,
        output_dir: str | Path = "exports",
        validate: bool = True,
    ) -> Path:
        """Export a trained checkpoint to the target mobile format.

        Args:
            checkpoint_path: Path to a .pth state dict from TrainingPipeline.
            format:          One of 'onnx', 'tflite', 'coreml'.
            output_dir:      Directory to write the exported model file.
            validate:        When True, verify exported model outputs match
                             PyTorch outputs within _OUTPUT_ATOL tolerance.

        Returns:
            Path to the exported model file.
        """
        checkpoint_path = Path(checkpoint_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if format == "onnx":
            return self._export_onnx(checkpoint_path, output_dir, validate)
        if format == "tflite":
            return self._export_tflite(checkpoint_path, output_dir, validate)
        if format == "coreml":
            return self._export_coreml(checkpoint_path, output_dir, validate)
        raise ValueError(f"Unknown format '{format}'. Choose from: onnx, tflite, coreml")

    # ------------------------------------------------------------------
    # ONNX
    # ------------------------------------------------------------------

    def _export_onnx(
        self, checkpoint_path: Path, output_dir: Path, validate: bool
    ) -> Path:
        import onnx
        import onnxruntime.quantization.quant_utils as _qu
        from onnxruntime.quantization import QuantType, quantize_dynamic

        classifier = RipenessClassifier(checkpoint_path=checkpoint_path, pretrained=False)
        model = classifier._model.cpu().eval()

        fp32_path = output_dir / "model_fp32.onnx"
        int8_path = output_dir / "model_int8.onnx"

        # Export FP32 ONNX (opset 18 — required by PyTorch 2.12 exporter)
        torch.onnx.export(
            model,
            (_DUMMY_INPUT,),
            str(fp32_path),
            input_names=["image"],
            output_names=["logits"],
            opset_version=18,
        )
        onnx.checker.check_model(str(fp32_path))

        # Quantize to int8. PyTorch 2.12's dynamo-based ONNX exporter produces
        # graphs whose shape metadata confuses the quantizer's built-in shape
        # inference pass. We patch load_model_with_shape_infer to skip that step
        # and pass DefaultTensorType so the quantizer can resolve weight types.
        _original_lm = _qu.load_model_with_shape_infer
        _qu.load_model_with_shape_infer = lambda p: onnx.load(str(p))
        try:
            quantize_dynamic(
                str(fp32_path),
                str(int8_path),
                weight_type=QuantType.QInt8,
                extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
            )
        finally:
            _qu.load_model_with_shape_infer = _original_lm

        if validate:
            self._validate_onnx(model, int8_path)

        # Remove intermediate FP32 file
        fp32_path.unlink(missing_ok=True)
        return int8_path

    def _validate_onnx(self, pytorch_model: torch.nn.Module, onnx_path: Path) -> None:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path))
        dummy = _DUMMY_INPUT.numpy()

        with torch.no_grad():
            pt_out = pytorch_model(_DUMMY_INPUT).numpy()

        ort_out = session.run(["logits"], {"image": dummy})[0]

        if not np.allclose(pt_out, ort_out, atol=_OUTPUT_ATOL):
            max_diff = np.abs(pt_out - ort_out).max()
            raise ValueError(
                f"Exported ONNX model output differs from PyTorch by {max_diff:.6f} "
                f"(tolerance {_OUTPUT_ATOL}). Export may be corrupted."
            )

    # ------------------------------------------------------------------
    # TFLite (requires tensorflow)
    # ------------------------------------------------------------------

    def _export_tflite(
        self, checkpoint_path: Path, output_dir: Path, validate: bool
    ) -> Path:
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            raise ImportError(
                "TFLite export requires TensorFlow. "
                "Install it with: pip install tensorflow\n"
                "Alternatively, export to ONNX and use ai.onnxruntime:onnxruntime-android "
                "for Android deployment."
            ) from None
        raise NotImplementedError("TFLite export is not yet implemented.")

    # ------------------------------------------------------------------
    # Core ML (requires coremltools)
    # ------------------------------------------------------------------

    def _export_coreml(
        self, checkpoint_path: Path, output_dir: Path, validate: bool
    ) -> Path:
        try:
            import coremltools  # noqa: F401
        except ImportError:
            raise ImportError(
                "Core ML export requires coremltools. "
                "Install it with: pip install coremltools\n"
                "Alternatively, export to ONNX and use onnxruntime-objc "
                "for iOS deployment."
            ) from None
        raise NotImplementedError("Core ML export is not yet implemented.")
