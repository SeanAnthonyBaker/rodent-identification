package uk.local.ratwatch.phone

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Build
import android.os.SystemClock
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

data class DetectionResult(
    val boxFrame: FloatArray, // [x1, y1, x2, y2]
    val confidence: Float,
    val inferenceTimeMs: Long,
    val inputSize: Int
)

/**
 * Optimized Single-Class YOLO11n INT8 Detector for Galaxy S21.
 * Auto-selects Snapdragon vs Exynos hardware delegate at startup.
 */
class YoloDetector(
    private val context: Context,
    var inputSize: Int = 416
) {
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var nnApiDelegate: NnApiDelegate? = null
    private val confSendThreshold = 0.28f

    init {
        initializeInterpreter()
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun initializeInterpreter() {
        val options = Interpreter.Options()
        val isSnapdragon = Build.HARDWARE.contains("qcom", ignoreCase = true) || Build.SOC_MODEL.contains("SM8350", ignoreCase = true)

        try {
            if (isSnapdragon) {
                // Snapdragon 888: Prefer NNAPI / QNN
                nnApiDelegate = NnApiDelegate(NnApiDelegate.Options().apply {
                    setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_LOW_LATENCY)
                    setAllowFp16(true)
                })
                options.addDelegate(nnApiDelegate)
            } else {
                // Exynos 2100: Prefer GPU Delegate (OpenCL)
                gpuDelegate = GpuDelegate(GpuDelegate.Options().apply {
                    setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER)
                })
                options.addDelegate(gpuDelegate)
            }
        } catch (e: Exception) {
            // Fallback to 2 CPU threads with XNNPACK
            options.setNumThreads(2)
            options.setUseXNNPACK(true)
        }

        val modelBuffer = loadModelFile("rat_yolo11n_int8.tflite")
        interpreter = Interpreter(modelBuffer, options)
    }

    /**
     * Executes forward inference on pre-resized bitmap or buffer.
     */
    fun detect(bitmap: Bitmap, origWidth: Int, origHeight: Int): DetectionResult? {
        val t0 = SystemClock.elapsedRealtime()
        val scaled = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        
        val inputBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3)
        inputBuffer.order(ByteOrder.nativeOrder())
        
        val intValues = IntArray(inputSize * inputSize)
        scaled.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)
        
        var pixel = 0
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val value = intValues[pixel++]
                inputBuffer.put(((value shr 16) and 0xFF).toByte())
                inputBuffer.put(((value shr 8) and 0xFF).toByte())
                inputBuffer.put((value and 0xFF).toByte())
            }
        }

        // YOLO11 1-class output format: [1, 5, 3549] -> [cx, cy, w, h, conf]
        val outputArray = Array(1) { Array(5) { FloatArray(3549) } }
        interpreter?.run(inputBuffer, outputArray)

        val inferTime = SystemClock.elapsedRealtime() - t0

        // Parse best detection
        var bestConf = 0.0f
        var bestBox = FloatArray(4)

        for (i in 0 until 3549) {
            val conf = outputArray[0][4][i]
            if (conf >= confSendThreshold && conf > bestConf) {
                bestConf = conf
                val cx = outputArray[0][0][i] * origWidth / inputSize
                val cy = outputArray[0][1][i] * origHeight / inputSize
                val w = outputArray[0][2][i] * origWidth / inputSize
                val h = outputArray[0][3][i] * origHeight / inputSize

                bestBox = floatArrayOf(
                    (cx - w / 2).coerceAtLeast(0f),
                    (cy - h / 2).coerceAtLeast(0f),
                    (cx + w / 2).coerceAtMost(origWidth.toFloat()),
                    (cy + h / 2).coerceAtMost(origHeight.toFloat())
                )
            }
        }

        if (bestConf >= confSendThreshold) {
            return DetectionResult(bestBox, bestConf, inferTime, inputSize)
        }

        return null
    }

    fun close() {
        interpreter?.close()
        gpuDelegate?.close()
        nnApiDelegate?.close()
    }
}
