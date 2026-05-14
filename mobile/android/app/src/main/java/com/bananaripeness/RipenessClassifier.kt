package com.bananaripeness

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import java.nio.FloatBuffer

private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
private val IMAGENET_STD  = floatArrayOf(0.229f, 0.224f, 0.225f)
private const val INPUT_SIZE = 224
private const val MODEL_ASSET = "model_int8.onnx"

enum class RipenessStage(
    val label: String,
    val description: String,
    val colorRes: Int,
) {
    UNRIPE("unripe", "Unripe — not ready yet", R.color.stage_unripe),
    NEARLY_RIPE("nearly-ripe", "Nearly ripe — a few more days", R.color.stage_nearly_ripe),
    RIPE("ripe", "Ripe — ready to sell", R.color.stage_ripe),
    OVERRIPE("overripe", "Overripe — sell immediately", R.color.stage_overripe);

    companion object {
        fun fromIndex(index: Int) = entries[index]
    }
}

data class Prediction(
    val stage: RipenessStage,
    val confidence: Float,
    val lowConfidence: Boolean,
)

class RipenessClassifier(context: Context) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val modelBytes = context.assets.open(MODEL_ASSET).readBytes()
        session = env.createSession(modelBytes, OrtSession.SessionOptions())
    }

    fun predict(bitmap: Bitmap): Prediction {
        val scaled = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val tensor = bitmapToTensor(scaled)

        val inputName = session.inputNames.first()
        val onnxTensor = OnnxTensor.createTensor(env, tensor,
            longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()))

        val output = session.run(mapOf(inputName to onnxTensor))
        val logits = (output[0].value as Array<FloatArray>)[0]

        val probs = softmax(logits)
        val topIdx = probs.indices.maxByOrNull { probs[it] }!!
        val confidence = probs[topIdx]

        onnxTensor.close()
        output.close()

        return Prediction(
            stage = RipenessStage.fromIndex(topIdx),
            confidence = confidence,
            lowConfidence = confidence < LOW_CONFIDENCE_THRESHOLD,
        )
    }

    private fun bitmapToTensor(bitmap: Bitmap): FloatBuffer {
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val buffer = FloatBuffer.allocate(3 * INPUT_SIZE * INPUT_SIZE)
        val r = FloatArray(INPUT_SIZE * INPUT_SIZE)
        val g = FloatArray(INPUT_SIZE * INPUT_SIZE)
        val b = FloatArray(INPUT_SIZE * INPUT_SIZE)

        for (i in pixels.indices) {
            val px = pixels[i]
            r[i] = ((px shr 16 and 0xFF) / 255f - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
            g[i] = ((px shr 8  and 0xFF) / 255f - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
            b[i] = ((px        and 0xFF) / 255f - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
        }
        buffer.put(r); buffer.put(g); buffer.put(b)
        buffer.rewind()
        return buffer
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.max()
        val exps = logits.map { Math.exp((it - max).toDouble()).toFloat() }
        val sum = exps.sum()
        return exps.map { it / sum }.toFloatArray()
    }

    override fun close() {
        session.close()
        env.close()
    }

    companion object {
        const val LOW_CONFIDENCE_THRESHOLD = 0.6f
    }
}
