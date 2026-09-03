package uk.local.ratwatch.phone

import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit
import kotlin.math.max
import kotlin.math.min

/**
 * Net Thread Worker: Dispatches 40% expanded crop to DGX Spark over LAN in <= 20 ms.
 */
class NetworkPoster(
    private var sparkUrl: String = "http://spark.local:8088/v1/sighting"
) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(800, TimeUnit.MILLISECONDS)
        .writeTimeout(2, TimeUnit.SECONDS)
        .readTimeout(3, TimeUnit.SECONDS)
        .retryOnConnectionFailure(true)
        .build()

    private var lastSendTime: Long = 0
    private var lastSentBox: FloatArray? = null

    /**
     * Builds a 40% expanded context crop, encodes JPEG q72 <=80KB, and POSTs multipart.
     */
    fun postCropAsync(
        fullBitmap: Bitmap,
        detection: DetectionResult,
        deviceId: String = "s21-garden",
        onVerdict: (String) -> Unit
    ) {
        val now = SystemClock.elapsedRealtime()

        // 4.0s debounce for same track
        if (now - lastSendTime < 4000 && lastSentBox != null) {
            val iou = calculateIoU(detection.boxFrame, lastSentBox!!)
            if (iou >= 0.15f) {
                return // Suppress duplicate sends of the same track
            }
        }

        lastSendTime = now
        lastSentBox = detection.boxFrame

        // 1. Expand box by 40% on all sides for tail/environment context
        val box = detection.boxFrame
        val origW = fullBitmap.width.toFloat()
        val origH = fullBitmap.height.toFloat()

        val boxW = box[2] - box[0]
        val boxH = box[3] - box[1]

        val padX = boxW * 0.40f
        val padY = boxH * 0.40f

        var cropX1 = max(0f, box[0] - padX)
        var cropY1 = max(0f, box[1] - padY)
        var cropX2 = min(origW, box[2] + padX)
        var cropY2 = min(origH, box[3] + padY)

        // Ensure minimum 160px short edge
        if ((cropX2 - cropX1) < 160f) {
            val diff = (160f - (cropX2 - cropX1)) / 2f
            cropX1 = max(0f, cropX1 - diff)
            cropX2 = min(origW, cropX2 + diff)
        }
        if ((cropY2 - cropY1) < 160f) {
            val diff = (160f - (cropY2 - cropY1)) / 2f
            cropY1 = max(0f, cropY1 - diff)
            cropY2 = min(origH, cropY2 + diff)
        }

        val cropW = (cropX2 - cropX1).toInt()
        val cropH = (cropY2 - cropY1).toInt()

        val croppedBitmap = Bitmap.createBitmap(fullBitmap, cropX1.toInt(), cropY1.toInt(), cropW, cropH)

        // 2. Scale down so longest edge <= 640 px
        val longestEdge = max(cropW, cropH)
        val finalBitmap = if (longestEdge > 640) {
            val scale = 640f / longestEdge.toFloat()
            Bitmap.createScaledBitmap(croppedBitmap, (cropW * scale).toInt(), (cropH * scale).toInt(), true)
        } else {
            croppedBitmap
        }

        // 3. Encode JPEG quality 72 (Target <= 80 KB)
        var stream = ByteArrayOutputStream()
        finalBitmap.compress(Bitmap.CompressFormat.JPEG, 72, stream)
        var jpegBytes = stream.toByteArray()

        if (jpegBytes.size > 80 * 1024) {
            stream = ByteArrayOutputStream()
            finalBitmap.compress(Bitmap.CompressFormat.JPEG, 60, stream)
            jpegBytes = stream.toByteArray()
        }

        // 4. Construct Metadata JSON
        val metaJson = JSONObject().apply {
            put("device_id", deviceId)
            put("ts_ms", System.currentTimeMillis())
            put("conf", detection.confidence.toDouble())
            put("box_frame", JSONArray(box.toList()))
            put("frame_wh", JSONArray(listOf(fullBitmap.width, fullBitmap.height)))
            put("yolo_input", detection.inputSize)
            put("infer_ms", detection.inferenceTimeMs)
            put("track_id", (System.currentTimeMillis() % 10000).toInt())
        }

        // 5. Construct Multipart Request Body
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "image",
                "crop.jpg",
                jpegBytes.toRequestBody("image/jpeg".toMediaTypeOrNull(), 0, jpegBytes.size)
            )
            .addFormDataPart("meta", metaJson.toString())
            .build()

        val request = Request.Builder()
            .url(sparkUrl)
            .post(requestBody)
            .build()

        // 6. Non-blocking Network Execution
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: java.io.IOException) {
                onVerdict("Spark connection failed: ${e.message}")
            }

            override fun onResponse(call: Call, response: Response) {
                response.use {
                    if (it.isSuccessful) {
                        val bodyStr = it.body?.string() ?: ""
                        onVerdict(bodyStr)
                    } else {
                        onVerdict("Spark HTTP ${it.code}")
                    }
                }
            }
        })
    }

    private fun calculateIoU(boxA: FloatArray, boxB: FloatArray): Float {
        val xA = max(boxA[0], boxB[0])
        val yA = max(boxA[1], boxB[1])
        val xB = min(boxA[2], boxB[2])
        val yB = min(boxA[3], boxB[3])

        val interArea = max(0f, xB - xA) * max(0f, yB - yA)
        val boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        val boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        val unionArea = boxAArea + boxBArea - interArea
        return if (unionArea > 0f) interArea / unionArea else 0f
    }
}
