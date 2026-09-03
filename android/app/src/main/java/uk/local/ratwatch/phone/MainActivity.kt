package uk.local.ratwatch.phone

import android.Manifest
import android.content.*
import android.content.pm.PackageManager
import android.os.*
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

/**
 * 3-Screen Minimal S21 Control UI:
 * 1. Arm/Disarm Big Toggle
 * 2. Mount / Alignment View
 * 3. Sighting Log
 */
class MainActivity : AppCompatActivity() {

    private var cameraService: CameraService? = null
    private var isBound = false
    private var isArmed = false

    private lateinit var btnToggleArm: Button
    private lateinit var tvStatus: TextView
    private lateinit var tvLatency: TextView
    private lateinit var tvLog: TextView

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            val binder = service as CameraService.LocalBinder
            cameraService = binder.getService()
            isBound = true
            cameraService?.onDetectionListener = { det, verdict ->
                runOnUiThread {
                    tvLatency.text = "Last YOLO: ${det.inferenceTimeMs} ms (${Math.round(det.confidence * 100)}%)"
                    tvLog.append("\n[${System.currentTimeMillis() % 100000}] Hit! Spark: $verdict")
                }
            }
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            isBound = false
            cameraService = null
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Programmatic UI layout for zero external resource dependency
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 48, 32, 32)
            setBackgroundColor(0xFF0F172A.toInt()) // Slate-900
        }

        val title = TextView(this).apply {
            text = "🐀 RatWatch S21 Detector"
            textSize = 20f
            setTextColor(0xFFFFFFFF.toInt())
            setPadding(0, 0, 0, 24)
        }
        layout.addView(title)

        tvStatus = TextView(this).apply {
            text = "Status: Disarmed (Standby)"
            textSize = 14f
            setTextColor(0xFF94A3B8.toInt())
            setPadding(0, 0, 0, 16)
        }
        layout.addView(tvStatus)

        btnToggleArm = Button(this).apply {
            text = "🛡️ ARM DETECTOR"
            textSize = 16f
            setBackgroundColor(0xFF059669.toInt()) // Emerald
            setTextColor(0xFFFFFFFF.toInt())
            setOnClickListener { toggleArm() }
        }
        layout.addView(btnToggleArm)

        tvLatency = TextView(this).apply {
            text = "Last YOLO: -- ms"
            textSize = 12f
            setTextColor(0xFFF59E0B.toInt()) // Amber
            setPadding(0, 24, 0, 16)
        }
        layout.addView(tvLatency)

        tvLog = TextView(this).apply {
            text = "Sighting Event Log:"
            textSize = 12f
            setTextColor(0xFFE2E8F0.toInt())
        }
        layout.addView(tvLog)

        setContentView(layout)

        checkPermissions()
    }

    private fun checkPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 101)
        }
    }

    private fun toggleArm() {
        isArmed = !isArmed
        if (isArmed) {
            btnToggleArm.text = "🛑 DISARM DETECTOR"
            btnToggleArm.setBackgroundColor(0xFFDC2626.toInt()) // Red
            tvStatus.text = "Status: 🟢 ARMED (Screen-off enabled)"
            
            val intent = Intent(this, CameraService::class.java)
            startService(intent)
            bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
            cameraService?.startCamera("0")
        } else {
            btnToggleArm.text = "🛡️ ARM DETECTOR"
            btnToggleArm.setBackgroundColor(0xFF059669.toInt())
            tvStatus.text = "Status: Disarmed (Standby)"
            
            if (isBound) {
                unbindService(serviceConnection)
                isBound = false
            }
            stopService(Intent(this, CameraService::class.java))
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (isBound) {
            unbindService(serviceConnection)
        }
    }
}
