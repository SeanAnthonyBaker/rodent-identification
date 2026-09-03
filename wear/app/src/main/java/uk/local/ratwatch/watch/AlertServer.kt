package uk.local.ratwatch.watch

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Base64
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStream
import java.net.ServerSocket
import java.net.Socket
import kotlin.concurrent.thread

/**
 * Lightweight embedded LAN alert server on Wear OS watch binding port 8099.
 */
class AlertServer(
    private val context: Context,
    private val port: Int = 8099,
    private val onAlertReceived: (String, Bitmap?) -> Unit
) {
    private var serverSocket: ServerSocket? = null
    private var isRunning = false
    private var lastAlertText = ""
    private var lastAlertTime = 0L

    fun start() {
        if (isRunning) return
        isRunning = true

        thread(name = "WearAlertServer") {
            try {
                serverSocket = ServerSocket(port)
                while (isRunning) {
                    val client = serverSocket?.accept() ?: break
                    handleClient(client)
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    private fun handleClient(client: Socket) {
        thread {
            try {
                val reader = BufferedReader(InputStreamReader(client.getInputStream()))
                var contentLength = 0
                var line: String?

                // Read headers
                while (reader.readLine().also { line = it } != null) {
                    if (line!!.isEmpty()) break
                    if (line!!.startsWith("Content-Length:", ignoreCase = true)) {
                        contentLength = line!!.substringAfter(":").trim().toInt()
                    }
                }

                // Read body
                val bodyChars = CharArray(contentLength)
                var readTotal = 0
                while (readTotal < contentLength) {
                    val read = reader.read(bodyChars, readTotal, contentLength - readTotal)
                    if (read == -1) break
                    readTotal += read
                }

                val bodyStr = String(bodyChars)
                val json = JSONObject(bodyStr)
                val watchText = json.optString("watch_text", "Rat Alert")
                val thumbB64 = json.optString("thumb_jpeg_b64", "")

                val now = System.currentTimeMillis()
                // 20s deduplication suppression
                if (watchText == lastAlertText && (now - lastAlertTime < 20000)) {
                    sendResponse(client.getOutputStream(), 200, "{\"ok\":true,\"suppressed\":true}")
                    client.close()
                    return@thread
                }

                lastAlertText = watchText
                lastAlertTime = now

                // Trigger 400ms haptic vibration
                vibrateWatch()

                // Decode thumbnail
                var bitmap: Bitmap? = null
                if (thumbB64.isNotEmpty()) {
                    val bytes = Base64.decode(thumbB64, Base64.DEFAULT)
                    bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                }

                onAlertReceived(watchText, bitmap)

                sendResponse(client.getOutputStream(), 200, "{\"ok\":true,\"shown_ms\":12}")
                client.close()

            } catch (e: Exception) {
                e.printStackTrace()
                try { client.close() } catch (_: Exception) {}
            }
        }
    }

    private fun sendResponse(out: OutputStream, code: Int, json: String) {
        val resp = "HTTP/1.1 $code OK\r\nContent-Type: application/json\r\nContent-Length: ${json.length}\r\nConnection: close\r\n\r\n$json"
        out.write(resp.toByteArray())
        out.flush()
    }

    private fun vibrateWatch() {
        val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vibratorManager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
            vibratorManager.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createOneShot(400, VibrationEffect.DEFAULT_AMPLITUDE))
        } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(400)
        }
    }

    fun stop() {
        isRunning = false
        try { serverSocket?.close() } catch (_: Exception) {}
    }
}
