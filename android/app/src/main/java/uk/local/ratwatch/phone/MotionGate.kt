package uk.local.ratwatch.phone

import java.nio.ByteBuffer
import kotlin.math.abs

/**
 * High-speed motion gate for S21 Camera2 pipeline.
 * Computes 160x90 grayscale absolute difference in <= 15 ms.
 */
class MotionGate(
    private val width: Int = 160,
    private val height: Int = 90,
    private val thresholdPct: Double = 0.04,
    private val diffPixelThreshold: Int = 25
) {
    private var prevFrame: ByteArray? = null
    private var wakeHoldCount: Int = 0

    /**
     * Checks if current downsampled grayscale buffer has material motion.
     * @param yuvData Full preview Y-plane or pre-sampled grayscale bytes
     * @return true if YOLO should run on this frame
     */
    fun hasMaterialMotion(grayData: ByteArray): Boolean {
        if (wakeHoldCount > 0) {
            wakeHoldCount--
            return true
        }

        val prev = prevFrame
        if (prev == null || prev.size != grayData.size) {
            prevFrame = grayData.clone()
            return false
        }

        var changedPixels = 0
        val totalPixels = grayData.size

        for (i in 0 until totalPixels) {
            val diff = abs((grayData[i].toInt() and 0xFF) - (prev[i].toInt() and 0xFF))
            if (diff > diffPixelThreshold) {
                changedPixels++
            }
        }

        System.arraycopy(grayData, 0, prev, 0, totalPixels)

        val deltaRatio = changedPixels.toDouble() / totalPixels.toDouble()
        if (deltaRatio >= thresholdPct) {
            // Keep YOLO awake for 8 consecutive frames after motion
            wakeHoldCount = 8
            return true
        }

        return false
    }
}
