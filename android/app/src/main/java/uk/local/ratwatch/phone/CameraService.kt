package uk.local.ratwatch.phone

import android.app.*
import android.content.Context
import android.content.Intent
import android.graphics.*
import android.hardware.camera2.*
import android.media.ImageReader
import android.os.*
import androidx.core.app.NotificationCompat
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

/**
 * Foreground Camera Service executing continuous detection on S21 even with screen off.
 */
class CameraService : Service() {

    private val binder = LocalBinder()
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null
    
    private var motionGate = MotionGate()
    private var yoloDetector: YoloDetector? = null
    private var networkPoster = NetworkPoster()

    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null

    private var wakeLock: PowerManager.WakeLock? = null
    var onDetectionListener: ((DetectionResult, String) -> Unit)? = null

    inner class LocalBinder : Binder() {
        fun getService(): CameraService = this@CameraService
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onCreate() {
        super.onCreate()
        acquireWakeLock()
        startForegroundNotification()
        startBackgroundThread()
        yoloDetector = YoloDetector(this, inputSize = 416)
    }

    private fun acquireWakeLock() {
        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "RatWatch:CameraWakeLock")
        wakeLock?.acquire(12 * 60 * 60 * 1000L) // 12 hours max
    }

    private fun startForegroundNotification() {
        val channelId = "ratwatch_s21_channel"
        val channelName = "RatWatch Detector Running"

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val chan = NotificationChannel(channelId, channelName, NotificationManager.IMPORTANCE_LOW)
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(chan)
        }

        val notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("RatWatch S21 Active")
            .setContentText("Camera2 & YOLO11n monitoring garden run...")
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setOngoing(true)
            .build()

        startForeground(1001, notification)
    }

    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("CameraBackground").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
    }

    fun startCamera(cameraId: String = "0") {
        val manager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try {
            imageReader = ImageReader.newInstance(1280, 720, ImageFormat.YUV_420_888, 2).apply {
                setOnImageAvailableListener({ reader ->
                    val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                    processFrame(image)
                    image.close()
                }, backgroundHandler)
            }

            manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraDevice = camera
                    createCaptureSession()
                }

                override fun onDisconnected(camera: CameraDevice) {
                    camera.close()
                    cameraDevice = null
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    camera.close()
                    cameraDevice = null
                }
            }, backgroundHandler)

        } catch (e: SecurityException) {
            e.printStackTrace()
        }
    }

    private fun createCaptureSession() {
        val device = cameraDevice ?: return
        val surface = imageReader?.surface ?: return

        device.createCaptureSession(listOf(surface), object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(session: CameraCaptureSession) {
                captureSession = session
                val requestBuilder = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                    addTarget(surface)
                    set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO)
                    set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                    set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
                    set(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE, CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF)
                }
                session.setRepeatingRequest(requestBuilder.build(), null, backgroundHandler)
            }

            override fun onConfigureFailed(session: CameraCaptureSession) {}
        }, backgroundHandler)
    }

    private fun processFrame(image: android.media.Image) {
        val yPlane = image.planes[0].buffer
        val yBytes = ByteArray(yPlane.remaining())
        yPlane.get(yBytes)

        // 1. Fast Motion Gate Check (<=15 ms)
        val graySample = sampleGrayscale(yBytes, image.width, image.height, 160, 90)
        if (!motionGate.hasMaterialMotion(graySample)) {
            return // Skip YOLO if garden scene is quiet
        }

        // 2. Convert YUV to full Bitmap for YOLO & crop
        val bitmap = yuvToBitmap(image) ?: return

        // 3. Run YOLO11n INT8
        val det = yoloDetector?.detect(bitmap, image.width, image.height)
        if (det != null) {
            // 4. Dispatch Crop POST to DGX Spark Net Thread
            networkPoster.postCropAsync(bitmap, det) { verdictStr ->
                onDetectionListener?.invoke(det, verdictStr)
            }
        }
    }

    private fun sampleGrayscale(yBytes: ByteArray, w: Int, h: Int, targetW: Int, targetH: Int): ByteArray {
        val out = ByteArray(targetW * targetH)
        val stepX = w / targetW
        val stepY = h / targetH
        var idx = 0
        for (y in 0 until targetH) {
            val srcY = y * stepY
            for (x in 0 until targetW) {
                val srcX = x * stepX
                out[idx++] = yBytes[srcY * w + srcX]
            }
        }
        return out
    }

    private fun yuvToBitmap(image: android.media.Image): Bitmap? {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 80, out)
        val jpegBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
    }

    override fun onDestroy() {
        super.onDestroy()
        captureSession?.close()
        cameraDevice?.close()
        imageReader?.close()
        yoloDetector?.close()
        backgroundThread?.quitSafely()
        if (wakeLock?.isHeld == true) {
            wakeLock?.release()
        }
    }
}
