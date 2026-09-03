package uk.local.ratwatch.watch

import android.app.Activity
import android.graphics.Bitmap
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.WindowManager
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView

class MainActivity : Activity() {

    private var alertServer: AlertServer? = null
    private lateinit var tvStatus: TextView
    private lateinit var tvLocation: TextView
    private lateinit var ivThumb: ImageView
    private val mainHandler = Handler(Looper.getMainLooper())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(16, 16, 16, 16)
            setBackgroundColor(0xFF000000.toInt())
        }

        tvStatus = TextView(this).apply {
            text = "🛡️ Watch Guard Active"
            textSize = 13f
            setTextColor(0xFF10B981.toInt()) // Emerald
            gravity = Gravity.CENTER
        }
        layout.addView(tvStatus)

        tvLocation = TextView(this).apply {
            text = "Listening on LAN:8099"
            textSize = 12f
            setTextColor(0xFFE2E8F0.toInt())
            gravity = Gravity.CENTER
            setPadding(0, 8, 0, 8)
        }
        layout.addView(tvLocation)

        ivThumb = ImageView(this).apply {
            layoutParams = LinearLayout.LayoutParams(180, 120).apply {
                gravity = Gravity.CENTER_HORIZONTAL
            }
            scaleType = ImageView.ScaleType.FIT_CENTER
        }
        layout.addView(ivThumb)

        setContentView(layout)

        startServer()
    }

    private fun startServer() {
        alertServer = AlertServer(this, port = 8099) { watchText, thumbBitmap ->
            mainHandler.post {
                tvStatus.text = "🚨 RAT CONFIRMED!"
                tvStatus.setTextColor(0xFFEF4444.toInt()) // Red
                tvLocation.text = watchText

                if (thumbBitmap != null) {
                    ivThumb.setImageBitmap(thumbBitmap)
                }

                // Auto-clear after 15 seconds
                mainHandler.removeCallbacksAndMessages(null)
                mainHandler.postDelayed({
                    tvStatus.text = "🛡️ Watch Guard Active"
                    tvStatus.setTextColor(0xFF10B981.toInt())
                    tvLocation.text = "Listening on LAN:8099"
                    ivThumb.setImageDrawable(null)
                }, 15000)
            }
        }
        alertServer?.start()
    }

    override fun onDestroy() {
        super.onDestroy()
        alertServer?.stop()
    }
}
